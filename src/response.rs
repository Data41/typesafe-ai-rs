//! Answers, and reading them back with typed keys.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Index;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, KeyError, ProtocolError};
use crate::question::{Key, Question, QuestionKind, QuestionSet};

/// Token usage for a request.
///
/// TypeSafe charges per input token; output tokens are free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[non_exhaustive]
pub struct Usage {
    /// Input tokens consumed.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens produced.
    #[serde(default)]
    pub output_tokens: u64,
}

/// The answer to a Noul (yes/no) question.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NoulAnswer {
    /// Probability that the answer is yes, from 0 to 1.
    ///
    /// Noul answers carry no `confidence`: the value *is* the probability, so
    /// how far it sits from 0.5 is the certainty signal. See
    /// [Confidence](https://docs.typesafe.ai/confidence).
    pub noul: f64,
}

/// The answer to a Choice question.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ChoiceAnswer {
    /// The highest-probability option.
    pub choice: String,
    /// Every option mapped to its probability. The values sum to 1.
    pub probabilities: HashMap<String, f64>,
    /// How certain the model is, derived from `probabilities`, from 0 to 1.
    pub confidence: f64,
}

impl ChoiceAnswer {
    /// The probability assigned to one option, or `None` if it wasn't an option.
    pub fn probability(&self, label: &str) -> Option<f64> {
        self.probabilities.get(label).copied()
    }

    /// Every option, most probable first.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut ranked: Vec<(&str, f64)> = self
            .probabilities
            .iter()
            .map(|(label, probability)| (label.as_str(), *probability))
            .collect();
        // Probabilities are never NaN in practice; total_cmp keeps this total
        // anyway, and the label tiebreak keeps the order stable.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranked
    }

    /// The chosen option, but only if the model was at least this certain.
    ///
    /// The shape TypeSafe's [confidence-gated
    /// routing](https://docs.typesafe.ai/patterns/confidence-routing) pattern
    /// asks for: act on the answer, or fall through to a human.
    ///
    /// ```
    /// # use typesafe::ChoiceAnswer;
    /// # fn demo(answer: &ChoiceAnswer) {
    /// match answer.choice_above(0.8) {
    ///     Some(team) => println!("routing to {team}"),
    ///     None => println!("too close to call; escalating"),
    /// }
    /// # }
    /// ```
    pub fn choice_above(&self, min_confidence: f64) -> Option<&str> {
        (self.confidence >= min_confidence).then_some(self.choice.as_str())
    }
}

/// The answer to a Score question.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ScoreAnswer {
    /// The probability-weighted score. May land between levels.
    pub score: f64,
    /// The rubric, indexed by level.
    ///
    /// The API sends this as an object keyed `"0"`, `"1"`, … which is why it
    /// is re-indexed here: iterating those keys as strings puts `"10"` between
    /// `"1"` and `"2"`.
    pub legend: Vec<String>,
    /// Probability per level, indexed by level. The values sum to 1.
    ///
    /// Empty when the API omits the field, which its own documented examples
    /// sometimes do.
    pub probabilities: Vec<f64>,
    /// How certain the model is, derived from `probabilities`, from 0 to 1.
    pub confidence: f64,
}

impl ScoreAnswer {
    /// How many levels the rubric has.
    pub fn levels(&self) -> usize {
        self.legend.len()
    }

    /// The rubric description for one level.
    pub fn legend_for(&self, level: usize) -> Option<&str> {
        self.legend.get(level).map(String::as_str)
    }

    /// The probability assigned to one level.
    pub fn probability(&self, level: usize) -> Option<f64> {
        self.probabilities.get(level).copied()
    }

    /// The score rounded to the nearest whole level, clamped to the rubric.
    ///
    /// Useful when downstream code needs a discrete bucket, but note that
    /// rounding throws away the between-levels detail that is the point of a
    /// Score.
    pub fn nearest_level(&self) -> usize {
        if self.legend.is_empty() {
            return 0;
        }
        let rounded = self.score.round();
        if rounded < 0.0 {
            return 0;
        }
        (rounded as usize).min(self.legend.len() - 1)
    }
}

/// One answer, of whichever type its question asked for.
///
/// Normally you reach an answer through a typed [`Key`], which skips this
/// enum entirely. It is what you get from the untyped escape hatches
/// ([`SystemOne::by_name`], [`SystemOne::iter`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Answer {
    /// A yes/no answer.
    Noul(NoulAnswer),
    /// A selected option with its distribution.
    Choice(ChoiceAnswer),
    /// A score with its rubric and distribution.
    Score(ScoreAnswer),
}

impl Answer {
    /// The wire tag for this answer's type.
    pub fn tag(&self) -> &'static str {
        match self {
            Answer::Noul(_) => "noul",
            Answer::Choice(_) => "choice",
            Answer::Score(_) => "score",
        }
    }

    /// The Noul answer, if that is what this is.
    pub fn as_noul(&self) -> Option<&NoulAnswer> {
        match self {
            Answer::Noul(answer) => Some(answer),
            _ => None,
        }
    }

    /// The Choice answer, if that is what this is.
    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Answer::Choice(answer) => Some(answer),
            _ => None,
        }
    }

    /// The Score answer, if that is what this is.
    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Answer::Score(answer) => Some(answer),
            _ => None,
        }
    }
}

/// The result of evaluating one state against a [`QuestionSet`].
///
/// Index it with the keys the question builders returned:
///
/// ```no_run
/// # use typesafe::{Client, QuestionSet};
/// # async fn demo() -> Result<(), typesafe::Error> {
/// # let client = Client::new()?;
/// let mut q = QuestionSet::new();
/// let urgent = q.noul("is_urgent", "Does this convey urgency?");
/// let dept = q.choice("department", "Which team?", [
///     ("billing", "Payments"),
///     ("technical", "Bugs and outages"),
/// ]);
///
/// let answers = client.system_one(&q).state("My payouts keep failing!").send().await?;
///
/// let is_urgent: f64 = answers[urgent].noul;
/// let team: &str = &answers[dept].choice;
/// # let _ = (is_urgent, team);
/// # Ok(())
/// # }
/// ```
///
/// Indexing cannot fail for a key from the set that produced the response:
/// every question is checked to have an answer of the right type before this
/// value is constructed. A key from a *different* set panics rather than
/// return a plausible number from the wrong question — use
/// [`get`](Self::get) where a panic is unacceptable.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SystemOne {
    model: String,
    usage: Usage,
    request_id: Option<String>,
    set_id: NonZeroU64,
    names: Arc<[String]>,
    answers: Vec<Answer>,
}

impl SystemOne {
    /// The versioned model that answered, e.g. `jev-1.13.0`.
    ///
    /// Worth logging: `jev-latest` is an alias that moves when a new release
    /// ships, and this reports what actually ran.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Token usage for the request.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The `x-typesafe-request-id` response header, when present.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// How many answers came back.
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    /// True when there are no answers.
    pub fn is_empty(&self) -> bool {
        self.answers.is_empty()
    }

    /// Resolve a typed key without panicking.
    ///
    /// The fallible twin of `[]`. Prefer indexing unless you are writing a
    /// library that must not panic.
    pub fn get<K: QuestionKind>(&self, key: Key<K>) -> Result<&K::Answer, KeyError> {
        if key.set_id() != self.set_id {
            return Err(KeyError::ForeignQuestionSet {
                key_set: key.set_id().get(),
                response_set: self.set_id.get(),
            });
        }

        let answer = self
            .answers
            .get(key.position() as usize)
            .ok_or(KeyError::OutOfRange {
                index: key.position(),
                len: self.answers.len(),
            })?;

        K::extract(answer).ok_or(KeyError::WrongType {
            expected: K::TAG,
            found: answer.tag(),
        })
    }

    /// Look an answer up by the name it was asked under.
    ///
    /// The untyped escape hatch, for when the question name is only known at
    /// runtime.
    pub fn by_name(&self, name: &str) -> Option<&Answer> {
        let index = self.names.iter().position(|candidate| candidate == name)?;
        self.answers.get(index)
    }

    /// Every answer with its question name, in the order the questions were
    /// asked.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Answer)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip(self.answers.iter())
    }

    /// Build a response, checking it against the questions that produced it.
    ///
    /// This is what licenses the panicking `Index` impl: every question must
    /// have an answer, and every answer must be the type its question asked
    /// for. Answers for names we did not ask about are ignored — they cannot
    /// be reached by a key, and rejecting them would break on a harmless
    /// server change.
    pub(crate) fn from_wire(
        wire: WireResponse,
        request_id: Option<String>,
        questions: &QuestionSet,
    ) -> Result<Self, Error> {
        let WireResponse {
            model,
            mut answers,
            usage,
        } = wire;

        let mut parsed = Vec::with_capacity(questions.len());

        for (name, question) in questions.questions() {
            let raw = answers
                .remove(name)
                .ok_or_else(|| ProtocolError::MissingAnswer {
                    name: name.to_owned(),
                })?;

            let answer = parse_answer(name, raw, request_id.as_deref())?;
            let expected = question.expected_tag();

            if answer.tag() != expected {
                return Err(ProtocolError::WrongAnswerType {
                    name: name.to_owned(),
                    expected,
                    found: answer.tag().to_owned(),
                }
                .into());
            }

            parsed.push(answer);
        }

        Ok(SystemOne {
            model,
            usage,
            request_id,
            set_id: questions.set_id(),
            names: questions.shared_names(),
            answers: parsed,
        })
    }
}

impl<K: QuestionKind> Index<Key<K>> for SystemOne {
    type Output = K::Answer;

    #[track_caller]
    fn index(&self, key: Key<K>) -> &Self::Output {
        match self.get(key) {
            Ok(answer) => answer,
            Err(error) => panic!("cannot read answer from this response: {error}"),
        }
    }
}

impl<'a> IntoIterator for &'a SystemOne {
    type Item = (&'a str, &'a Answer);
    type IntoIter = std::iter::Zip<
        std::iter::Map<std::slice::Iter<'a, String>, fn(&String) -> &str>,
        std::slice::Iter<'a, Answer>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        let as_str: fn(&String) -> &str = String::as_str;
        self.names.iter().map(as_str).zip(self.answers.iter())
    }
}

impl Question {
    fn expected_tag(&self) -> &'static str {
        self.tag()
    }
}

/// The `POST /v1/systemone` response body, before validation.
#[derive(Debug, Deserialize)]
pub(crate) struct WireResponse {
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) answers: HashMap<String, Value>,
    #[serde(default)]
    pub(crate) usage: Usage,
}

#[derive(Deserialize)]
struct WireNoul {
    noul: f64,
}

#[derive(Deserialize)]
struct WireChoice {
    choice: String,
    #[serde(default)]
    probabilities: HashMap<String, f64>,
    #[serde(default)]
    confidence: f64,
}

#[derive(Deserialize)]
struct WireScore {
    score: f64,
    #[serde(default)]
    legend: HashMap<String, String>,
    #[serde(default)]
    probabilities: HashMap<String, f64>,
    #[serde(default)]
    confidence: f64,
}

/// Dispatch on the `type` discriminant by hand rather than with an internally
/// tagged enum, so an unrecognised type names itself in the error instead of
/// collapsing into an opaque "unknown variant".
fn parse_answer(name: &str, raw: Value, request_id: Option<&str>) -> Result<Answer, Error> {
    let tag = raw
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let decode = |source: serde_json::Error| Error::Decode {
        source,
        request_id: request_id.map(str::to_owned),
    };

    match tag.as_str() {
        "noul" => {
            let wire: WireNoul = serde_json::from_value(raw).map_err(decode)?;
            Ok(Answer::Noul(NoulAnswer { noul: wire.noul }))
        }
        "choice" => {
            let wire: WireChoice = serde_json::from_value(raw).map_err(decode)?;
            Ok(Answer::Choice(ChoiceAnswer {
                choice: wire.choice,
                probabilities: wire.probabilities,
                confidence: wire.confidence,
            }))
        }
        "score" => {
            let wire: WireScore = serde_json::from_value(raw).map_err(decode)?;
            let legend = level_indexed(name, wire.legend)?;
            let probabilities = level_indexed(name, wire.probabilities)?;

            if !probabilities.is_empty() && !legend.is_empty() && probabilities.len() != legend.len()
            {
                return Err(ProtocolError::MalformedScoreKeys {
                    name: name.to_owned(),
                    reason: format!(
                        "legend has {} level(s) but probabilities has {}",
                        legend.len(),
                        probabilities.len()
                    ),
                }
                .into());
            }

            Ok(Answer::Score(ScoreAnswer {
                score: wire.score,
                legend,
                probabilities,
                confidence: wire.confidence,
            }))
        }
        other => Err(ProtocolError::WrongAnswerType {
            name: name.to_owned(),
            expected: "noul, choice or score",
            found: if other.is_empty() {
                "an answer with no type field".to_owned()
            } else {
                other.to_owned()
            },
        }
        .into()),
    }
}

/// Turn a `{"0": _, "1": _}` object into a `Vec` indexed by level.
///
/// Rejects anything that is not a contiguous run starting at zero, because a
/// gap would silently shift every level after it.
fn level_indexed<T>(name: &str, map: HashMap<String, T>) -> Result<Vec<T>, ProtocolError> {
    if map.is_empty() {
        return Ok(Vec::new());
    }

    let malformed = |reason: String| ProtocolError::MalformedScoreKeys {
        name: name.to_owned(),
        reason,
    };

    let mut items: Vec<(usize, T)> = map
        .into_iter()
        .map(|(key, value)| {
            key.parse::<usize>()
                .map(|level| (level, value))
                .map_err(|_| malformed(format!("level key {key:?} is not a non-negative integer")))
        })
        .collect::<Result<_, _>>()?;

    items.sort_by_key(|(level, _)| *level);

    for (position, (level, _)) in items.iter().enumerate() {
        if position != *level {
            return Err(malformed(format!(
                "levels must run from 0 with no gaps; expected {position} but found {level}"
            )));
        }
    }

    Ok(items.into_iter().map(|(_, value)| value).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn levels_are_ordered_numerically_not_lexicographically() {
        // The bug this guards: as strings, "10" sorts between "1" and "2".
        let map: HashMap<String, String> = (0..12)
            .map(|level| (level.to_string(), format!("level {level}")))
            .collect();
        let ordered = level_indexed("q", map).unwrap();
        assert_eq!(ordered.len(), 12);
        for (level, description) in ordered.iter().enumerate() {
            assert_eq!(description, &format!("level {level}"));
        }
    }

    #[test]
    fn gapped_levels_are_rejected() {
        let map = HashMap::from([("0".to_owned(), "a"), ("2".to_owned(), "c")]);
        assert!(matches!(
            level_indexed("q", map),
            Err(ProtocolError::MalformedScoreKeys { .. })
        ));
    }

    #[test]
    fn non_numeric_levels_are_rejected() {
        let map = HashMap::from([("low".to_owned(), "a")]);
        assert!(matches!(
            level_indexed("q", map),
            Err(ProtocolError::MalformedScoreKeys { .. })
        ));
    }

    #[test]
    fn levels_not_starting_at_zero_are_rejected() {
        let map = HashMap::from([("1".to_owned(), "a"), ("2".to_owned(), "b")]);
        assert!(matches!(
            level_indexed("q", map),
            Err(ProtocolError::MalformedScoreKeys { .. })
        ));
    }

    #[test]
    fn a_score_answer_without_probabilities_still_parses() {
        // The quickstart in TypeSafe's own docs omits this field even though
        // the API reference marks it required.
        let answer = parse_answer(
            "frustration",
            json!({
                "type": "score",
                "score": 1.035,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "confidence": 0.842
            }),
            None,
        )
        .unwrap();

        let score = answer.as_score().unwrap();
        assert_eq!(score.score, 1.035);
        assert_eq!(score.levels(), 3);
        assert!(score.probabilities.is_empty());
        assert_eq!(score.legend_for(2), Some("Very angry"));
        assert_eq!(score.nearest_level(), 1);
    }

    #[test]
    fn an_unknown_answer_type_names_itself() {
        let error = parse_answer("q", json!({"type": "rank", "rank": 1}), None).unwrap_err();
        assert!(error.to_string().contains("rank"), "{error}");
    }

    #[test]
    fn mismatched_legend_and_probability_lengths_are_rejected() {
        let error = parse_answer(
            "q",
            json!({
                "type": "score",
                "score": 1.0,
                "legend": {"0": "a", "1": "b", "2": "c"},
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.5
            }),
            None,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Protocol(_)), "{error}");
    }

    #[test]
    fn ranked_options_come_back_most_probable_first() {
        let answer = ChoiceAnswer {
            choice: "technical".to_owned(),
            probabilities: HashMap::from([
                ("billing".to_owned(), 0.08),
                ("technical".to_owned(), 0.85),
                ("sales".to_owned(), 0.07),
            ]),
            confidence: 0.82,
        };
        let ranked = answer.ranked();
        assert_eq!(ranked[0].0, "technical");
        assert_eq!(ranked[2].0, "sales");
        assert_eq!(answer.probability("billing"), Some(0.08));
        assert_eq!(answer.choice_above(0.9), None);
        assert_eq!(answer.choice_above(0.8), Some("technical"));
    }

    #[test]
    fn nearest_level_clamps_to_the_rubric() {
        let answer = ScoreAnswer {
            score: 9.5,
            legend: vec!["a".to_owned(), "b".to_owned()],
            probabilities: Vec::new(),
            confidence: 0.5,
        };
        assert_eq!(answer.nearest_level(), 1);
    }
}
