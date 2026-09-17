//! Building a set of typed questions, and the keys that read their answers.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::entry::{Entry, IntoEntry};
use crate::error::QuestionSetError;
use crate::response::Answer;
use crate::sealed::Sealed;

/// Hands out a fresh identifier per `QuestionSet` so a key can tell whether it
/// belongs to the response it is being used against. Starts at 1 so the id can
/// be a `NonZeroU64`.
static NEXT_SET_ID: AtomicU64 = AtomicU64::new(1);

fn next_set_id() -> NonZeroU64 {
    let raw = NEXT_SET_ID.fetch_add(1, Ordering::Relaxed);
    NonZeroU64::new(raw).expect("QuestionSet id counter overflowed u64")
}

/// The kind of a question, linking it to the type of answer it produces.
///
/// Sealed: the three implementors ([`Noul`], [`Choice`], [`Score`]) are the
/// three question types the API defines.
pub trait QuestionKind: Sealed + 'static {
    /// The answer type this kind of question produces.
    type Answer;

    /// The wire tag, used in error messages.
    const TAG: &'static str;

    /// Pull this kind's answer out of the generic enum, if it is that kind.
    #[doc(hidden)]
    fn extract(answer: &Answer) -> Option<&Self::Answer>;
}

macro_rules! question_kind {
    ($($marker:ident => $answer:ty, $tag:literal, $variant:ident);* $(;)?) => {
        $(
            #[doc = concat!("Marker type for ", $tag, " questions. See [`QuestionKind`].")]
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub struct $marker;

            impl Sealed for $marker {}

            impl QuestionKind for $marker {
                type Answer = $answer;
                const TAG: &'static str = $tag;

                fn extract(answer: &Answer) -> Option<&Self::Answer> {
                    match answer {
                        Answer::$variant(inner) => Some(inner),
                        _ => None,
                    }
                }
            }
        )*
    };
}

question_kind! {
    Noul => crate::response::NoulAnswer, "noul", Noul;
    Choice => crate::response::ChoiceAnswer, "choice", Choice;
    Score => crate::response::ScoreAnswer, "score", Score;
}

/// A handle to one question, carrying that question's type.
///
/// Returned by the [`QuestionSet`] builders and used to index a
/// [`SystemOne`](crate::SystemOne) response. Because the key remembers what
/// kind of question it came from, reading the answer needs no `match` and no
/// `unwrap`.
///
/// Keys are `Copy`, hold no borrow of the set, and stay valid for as long as
/// the set exists — so building one set and reusing it across many calls (the
/// [fan-out pattern](https://docs.typesafe.ai/patterns/fan-out)) works, and
/// stashing keys in a struct alongside the set works too.
///
/// See the type aliases [`NoulKey`], [`ChoiceKey`], and [`ScoreKey`].
pub struct Key<K: QuestionKind> {
    set: NonZeroU64,
    index: u32,
    // `fn() -> K` rather than `K` so the key stays Send + Sync + covariant
    // regardless of the marker type.
    kind: PhantomData<fn() -> K>,
}

/// A handle to a Noul (yes/no) question.
pub type NoulKey = Key<Noul>;
/// A handle to a Choice question.
pub type ChoiceKey = Key<Choice>;
/// A handle to a Score question.
pub type ScoreKey = Key<Score>;

impl<K: QuestionKind> Key<K> {
    /// This key's position within its question set.
    pub fn position(&self) -> u32 {
        self.index
    }

    pub(crate) fn set_id(&self) -> NonZeroU64 {
        self.set
    }
}

// Hand-written rather than derived: `#[derive]` would add a spurious `K: Copy`
// bound, and the marker types are never actually stored.
impl<K: QuestionKind> Clone for Key<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: QuestionKind> Copy for Key<K> {}

impl<K: QuestionKind> fmt::Debug for Key<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Key")
            .field("kind", &K::TAG)
            .field("set", &self.set)
            .field("index", &self.index)
            .finish()
    }
}

impl<K: QuestionKind> PartialEq for Key<K> {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set && self.index == other.index
    }
}

impl<K: QuestionKind> Eq for Key<K> {}

impl<K: QuestionKind> Hash for Key<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.set.hash(state);
        self.index.hash(state);
    }
}

/// Optional descriptions of what a yes and a no mean for a Noul question.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NoulCriteria {
    pub(crate) yes: Entry,
    pub(crate) no: Entry,
}

/// One question, in the shape the request body needs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Question {
    Noul {
        instructions: Entry,
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: Entry,
        criteria: Vec<(String, Entry)>,
    },
    Score {
        instructions: Entry,
        criteria: Vec<Entry>,
    },
}

impl Question {
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }
}

/// A named set of typed questions, evaluated together against one state.
///
/// Every question in a set is answered in a single API call, in parallel.
/// TypeSafe's docs are emphatic that this is the way to use the API — "asking
/// a question you might not need is close to free" — so a set is meant to be
/// built once and reused.
///
/// # Building
///
/// Each builder returns a typed [`Key`]. Hold onto it; it is how you read the
/// answer back.
///
/// ```
/// use typesafe::QuestionSet;
///
/// let mut q = QuestionSet::new();
/// let urgent = q.noul("is_urgent", "Does this convey urgency?");
/// let dept = q.choice("department", "Which team should handle this?", [
///     ("billing", "Payments, invoicing, refunds"),
///     ("technical", "Bugs, outages, integrations"),
/// ]);
/// let frustration = q.score("frustration", "How frustrated is the customer?", [
///     "Calm",
///     "Frustrated",
///     "Very angry",
/// ]);
/// assert_eq!(q.len(), 3);
/// # let _ = (urgent, dept, frustration);
/// ```
///
/// # Append-only, by design
///
/// Keys are positional, so a set only ever grows — there is no `remove` or
/// `clear`, and there never will be. Pushing never invalidates a key handed
/// out earlier.
///
/// For the same reason `QuestionSet` deliberately implements neither `Clone`,
/// `Default`, nor `Deserialize`: each would mint a set sharing another set's
/// identity, and a key from one would then silently resolve against the other.
/// Wrap the set in an [`Arc`](std::sync::Arc) to share it — that also makes
/// the append-only rule unbreakable, since `&mut self` becomes unreachable.
///
/// # Errors
///
/// The builders cannot fail, because they have to return a key. Problems
/// (a duplicate name, an empty option list) are recorded and reported by
/// [`validate`](Self::validate), and by `send()` before anything is sent.
pub struct QuestionSet {
    set_id: NonZeroU64,
    names: Vec<String>,
    questions: Vec<Question>,
    first_error: Option<QuestionSetError>,
    /// Shared with each response so `by_name` lookups need no re-allocation.
    /// Reset whenever a question is appended.
    names_cache: OnceLock<Arc<[String]>>,
}

impl QuestionSet {
    /// An empty question set.
    ///
    /// Not a `Default` impl: `Default` would give every set the same identity,
    /// which is exactly the confusion [`Key`] exists to prevent.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        QuestionSet {
            set_id: next_set_id(),
            names: Vec::new(),
            questions: Vec::new(),
            first_error: None,
            names_cache: OnceLock::new(),
        }
    }

    /// Ask a yes/no question.
    ///
    /// The answer is the probability that the answer is yes, from 0 to 1.
    ///
    /// ```
    /// # use typesafe::QuestionSet;
    /// let mut q = QuestionSet::new();
    /// let urgent = q.noul("is_urgent", "Does this convey urgency?");
    /// # let _ = urgent;
    /// ```
    pub fn noul<N, E>(&mut self, name: N, instructions: E) -> NoulKey
    where
        N: Into<String>,
        E: IntoEntry,
    {
        self.push(
            name.into(),
            Question::Noul {
                instructions: instructions.into_entry(),
                criteria: None,
            },
        )
    }

    /// Ask a yes/no question, spelling out what a yes and a no mean.
    ///
    /// Worth reaching for when the boundary is subtle. Pass `()` for a side
    /// you do not want to describe.
    ///
    /// ```
    /// # use typesafe::QuestionSet;
    /// let mut q = QuestionSet::new();
    /// let urgent = q.noul_with(
    ///     "is_urgent",
    ///     "Does this convey urgency?",
    ///     "Explicitly time-sensitive",
    ///     "No urgency expressed",
    /// );
    /// # let _ = urgent;
    /// ```
    pub fn noul_with<N, E, Y, F>(&mut self, name: N, instructions: E, yes: Y, no: F) -> NoulKey
    where
        N: Into<String>,
        E: IntoEntry,
        Y: IntoEntry,
        F: IntoEntry,
    {
        self.push(
            name.into(),
            Question::Noul {
                instructions: instructions.into_entry(),
                criteria: Some(NoulCriteria {
                    yes: yes.into_entry(),
                    no: no.into_entry(),
                }),
            },
        )
    }

    /// Ask the model to pick one option from a set you define.
    ///
    /// `criteria` is any iterator of `(label, description)`. A description can
    /// be a string, structured JSON, or null — and when you need a *mix* of
    /// described and undescribed options, build them with
    /// [`ChoiceCriteria`](crate::ChoiceCriteria), which slots into this same
    /// parameter.
    ///
    /// ```
    /// # use typesafe::QuestionSet;
    /// let mut q = QuestionSet::new();
    /// let dept = q.choice("department", "Which team should handle this?", [
    ///     ("billing", "Payments, invoicing, refunds"),
    ///     ("technical", "Bugs, outages, integrations"),
    ///     ("sales", "Pricing, upgrades, new accounts"),
    /// ]);
    /// # let _ = dept;
    /// ```
    pub fn choice<N, E, I, L, D>(&mut self, name: N, instructions: E, criteria: I) -> ChoiceKey
    where
        N: Into<String>,
        E: IntoEntry,
        I: IntoIterator<Item = (L, D)>,
        L: Into<String>,
        D: IntoEntry,
    {
        let name = name.into();
        let criteria: Vec<(String, Entry)> = criteria
            .into_iter()
            .map(|(label, description)| (label.into(), description.into_entry()))
            .collect();

        if criteria.is_empty() {
            self.record(QuestionSetError::EmptyChoiceCriteria { name: name.clone() });
        }

        self.push(
            name,
            Question::Choice {
                instructions: instructions.into_entry(),
                criteria,
            },
        )
    }

    /// Ask the model to pick one option, with no descriptions.
    ///
    /// The separate method is forced by coherence: a single trait covering
    /// both `[L; N]` and `[(L, D); N]` cannot be written, because `L` could
    /// itself be a tuple.
    ///
    /// ```
    /// # use typesafe::QuestionSet;
    /// let mut q = QuestionSet::new();
    /// let sentiment = q.choice_labels(
    ///     "sentiment",
    ///     "How does this read?",
    ///     ["positive", "neutral", "negative"],
    /// );
    /// # let _ = sentiment;
    /// ```
    pub fn choice_labels<N, E, I, L>(&mut self, name: N, instructions: E, labels: I) -> ChoiceKey
    where
        N: Into<String>,
        E: IntoEntry,
        I: IntoIterator<Item = L>,
        L: Into<String>,
    {
        self.choice(
            name,
            instructions,
            labels.into_iter().map(|label| (label, Entry::null())),
        )
    }

    /// Rate the state against an ordered rubric.
    ///
    /// Levels are indexed from zero in the order given. The answer is a
    /// probability-weighted value that can land between levels.
    ///
    /// The API requires at least two levels, and this signature enforces that
    /// **at compile time** — `q.score("x", "y", ["only one"])` does not build.
    /// For a rubric assembled at runtime, use
    /// [`score_iter`](Self::score_iter).
    ///
    /// # Stricter than the server
    ///
    /// The [API reference](https://docs.typesafe.ai/api) says a Score "must
    /// include at least two levels", but the live endpoint accepts a
    /// single-level rubric and answers `200`. It is not worth sending: the
    /// answer is degenerate by construction — score `0.0` and confidence
    /// `1.0`, every time, since there is nothing to distribute probability
    /// over. This crate rejects it rather than let you pay tokens for a
    /// foregone conclusion.
    ///
    /// ```
    /// # use typesafe::QuestionSet;
    /// let mut q = QuestionSet::new();
    /// let frustration = q.score("frustration", "How frustrated is the customer?", [
    ///     "Calm, just stating facts",
    ///     "Frustrated but civil",
    ///     "Very angry, strong language",
    /// ]);
    /// # let _ = frustration;
    /// ```
    pub fn score<N, E, D, const M: usize>(
        &mut self,
        name: N,
        instructions: E,
        criteria: [D; M],
    ) -> ScoreKey
    where
        N: Into<String>,
        E: IntoEntry,
        D: IntoEntry,
    {
        const {
            assert!(
                M >= 2,
                "a score question needs at least 2 levels; use score_iter for a rubric built at runtime"
            );
        }
        self.score_iter(name, instructions, criteria)
    }

    /// Rate the state against a rubric assembled at runtime.
    ///
    /// The two-level minimum is checked when the set is validated, rather than
    /// at compile time as in [`score`](Self::score) — see the note there on
    /// why this crate is stricter than the server.
    pub fn score_iter<N, E, I, D>(&mut self, name: N, instructions: E, criteria: I) -> ScoreKey
    where
        N: Into<String>,
        E: IntoEntry,
        I: IntoIterator<Item = D>,
        D: IntoEntry,
    {
        let name = name.into();
        let criteria: Vec<Entry> = criteria.into_iter().map(IntoEntry::into_entry).collect();

        if criteria.len() < 2 {
            self.record(QuestionSetError::TooFewScoreCriteria {
                name: name.clone(),
                len: criteria.len(),
            });
        }

        self.push(
            name,
            Question::Score {
                instructions: instructions.into_entry(),
                criteria,
            },
        )
    }

    /// How many questions the set holds.
    pub fn len(&self) -> usize {
        self.questions.len()
    }

    /// True when no questions have been added.
    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    /// The question names, in the order they were added.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// Check the set without sending it.
    ///
    /// `send()` runs this first, so calling it yourself is only needed when
    /// you want to fail early — validating a set assembled from configuration,
    /// say, at startup rather than on the first request.
    pub fn validate(&self) -> Result<(), QuestionSetError> {
        if let Some(error) = &self.first_error {
            return Err(error.clone());
        }
        if self.questions.is_empty() {
            return Err(QuestionSetError::Empty);
        }
        Ok(())
    }

    pub(crate) fn set_id(&self) -> NonZeroU64 {
        self.set_id
    }

    pub(crate) fn questions(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip(self.questions.iter())
    }

    /// The names as a shared slice, built once per set and handed to every
    /// response it produces.
    pub(crate) fn shared_names(&self) -> Arc<[String]> {
        self.names_cache
            .get_or_init(|| Arc::from(self.names.as_slice()))
            .clone()
    }

    fn record(&mut self, error: QuestionSetError) {
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }

    fn push<K: QuestionKind>(&mut self, name: String, question: Question) -> Key<K> {
        if self.names.contains(&name) {
            self.record(QuestionSetError::DuplicateName { name: name.clone() });
        }

        let index = u32::try_from(self.questions.len())
            .expect("a question set cannot hold more than u32::MAX questions");

        self.names.push(name);
        self.questions.push(question);
        // Appending invalidates the shared slice.
        self.names_cache = OnceLock::new();

        Key {
            set: self.set_id,
            index,
            kind: PhantomData,
        }
    }
}

impl fmt::Debug for QuestionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuestionSet")
            .field("set_id", &self.set_id)
            .field(
                "questions",
                &self
                    .questions()
                    .map(|(name, question)| (name, question.tag()))
                    .collect::<Vec<_>>(),
            )
            .field("error", &self.first_error)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_handed_out_in_insertion_order() {
        let mut q = QuestionSet::new();
        let a = q.noul("a", "?");
        let b = q.choice("b", "?", [("x", "X")]);
        assert_eq!(a.position(), 0);
        assert_eq!(b.position(), 1);
    }

    #[test]
    fn distinct_sets_get_distinct_ids() {
        let a = QuestionSet::new();
        let b = QuestionSet::new();
        assert_ne!(a.set_id(), b.set_id());
    }

    #[test]
    fn empty_set_is_rejected() {
        assert_eq!(QuestionSet::new().validate(), Err(QuestionSetError::Empty));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut q = QuestionSet::new();
        q.noul("dupe", "?");
        q.noul("dupe", "?");
        assert_eq!(
            q.validate(),
            Err(QuestionSetError::DuplicateName {
                name: "dupe".to_owned()
            })
        );
    }

    #[test]
    fn empty_choice_options_are_rejected() {
        let mut q = QuestionSet::new();
        q.choice("department", "?", Vec::<(String, String)>::new());
        assert_eq!(
            q.validate(),
            Err(QuestionSetError::EmptyChoiceCriteria {
                name: "department".to_owned()
            })
        );
    }

    #[test]
    fn score_iter_rejects_fewer_than_two_levels() {
        let mut q = QuestionSet::new();
        q.score_iter("frustration", "?", ["only one"]);
        assert_eq!(
            q.validate(),
            Err(QuestionSetError::TooFewScoreCriteria {
                name: "frustration".to_owned(),
                len: 1,
            })
        );
    }

    #[test]
    fn the_first_error_is_the_one_reported() {
        let mut q = QuestionSet::new();
        q.score_iter("a", "?", ["one"]);
        q.choice("b", "?", Vec::<(String, String)>::new());
        assert!(matches!(
            q.validate(),
            Err(QuestionSetError::TooFewScoreCriteria { .. })
        ));
    }

    #[test]
    fn a_valid_set_validates() {
        let mut q = QuestionSet::new();
        q.noul("a", "?");
        q.score("b", "?", ["low", "high"]);
        assert_eq!(q.validate(), Ok(()));
    }

    #[test]
    fn appending_refreshes_the_shared_names() {
        let mut q = QuestionSet::new();
        q.noul("a", "?");
        assert_eq!(&*q.shared_names(), ["a".to_owned()]);
        q.noul("b", "?");
        assert_eq!(&*q.shared_names(), ["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn shared_names_are_reused_between_calls() {
        let mut q = QuestionSet::new();
        q.noul("a", "?");
        assert!(Arc::ptr_eq(&q.shared_names(), &q.shared_names()));
    }

    #[test]
    fn keys_are_copy_and_outlive_borrows_of_the_set() {
        let mut q = QuestionSet::new();
        let key = q.noul("a", "?");
        q.noul("b", "?");
        let copied = key;
        assert_eq!(copied.position(), 0);
    }
}
