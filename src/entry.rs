//! Values accepted wherever the API takes an `EntryType`.
//!
//! The TypeSafe API accepts text, a JSON object, a JSON array, or null for
//! `state`, for every `instructions` field, and for every criterion
//! description. The [advanced structure][adv] docs lean on this heavily —
//! a Choice option's description can be an entire taxonomy subtree.
//!
//! [adv]: https://docs.typesafe.ai/primitives/advanced

use serde_json::Value;

use crate::sealed::Sealed;

/// Text, a JSON object, a JSON array, or null.
///
/// Rarely named directly — [`IntoEntry`] converts the common cases for you.
/// Reach for [`Entry::json`] when you have a `#[derive(Serialize)]` type, and
/// [`Entry::null`] when you want an explicit null.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Entry(Value);

impl Entry {
    /// An explicit JSON null, which leaves a criterion undescribed.
    pub fn null() -> Self {
        Entry(Value::Null)
    }

    /// Build an entry from any `Serialize` type.
    ///
    /// This is the escape hatch for your own structs. It is fallible because
    /// serialization can fail (a map with non-string keys, a `NaN`, a custom
    /// `Serialize` that errors).
    pub fn json<T>(value: &T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize + ?Sized,
    {
        serde_json::to_value(value).map(Entry)
    }

    /// True when this entry is JSON null.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Borrow the underlying JSON value.
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// Consume the entry and return the underlying JSON value.
    pub fn into_value(self) -> Value {
        self.0
    }
}

impl From<Entry> for Value {
    fn from(entry: Entry) -> Self {
        entry.0
    }
}

/// Conversion into an [`Entry`].
///
/// This is a sealed trait rather than a `From`/`Into<Value>` bound for two
/// reasons. A blanket `impl<T: Serialize> From<T> for Entry` is impossible —
/// it collides with core's reflexive `impl<T> From<T> for T`. And accepting
/// `Into<Value>` instead would leave anyone with a `#[derive(Serialize)]`
/// type writing `serde_json::to_value(..).unwrap()` at every call site.
///
/// Implemented for `&str`, `String`, `&String`, `Cow<str>`, [`Entry`],
/// [`serde_json::Value`], `Option<T>` (where `None` is null), and `()`
/// (also null).
pub trait IntoEntry: Sealed {
    /// Perform the conversion.
    fn into_entry(self) -> Entry;
}

macro_rules! into_entry {
    ($($ty:ty => |$v:ident| $body:expr),* $(,)?) => {
        $(
            impl Sealed for $ty {}
            impl IntoEntry for $ty {
                fn into_entry(self) -> Entry {
                    let $v = self;
                    $body
                }
            }
        )*
    };
}

into_entry! {
    Entry => |v| v,
    Value => |v| Entry(v),
    &str => |v| Entry(Value::String(v.to_owned())),
    String => |v| Entry(Value::String(v)),
    &String => |v| Entry(Value::String(v.clone())),
    std::borrow::Cow<'_, str> => |v| Entry(Value::String(v.into_owned())),
    () => |_v| Entry(Value::Null),
}

impl<T: IntoEntry> Sealed for Option<T> {}

impl<T: IntoEntry> IntoEntry for Option<T> {
    fn into_entry(self) -> Entry {
        match self {
            Some(value) => value.into_entry(),
            None => Entry::null(),
        }
    }
}

/// Choice options with per-option descriptions, for when an array of tuples
/// will not do.
///
/// [`QuestionSet::choice`] takes any `IntoIterator<Item = (label, description)>`,
/// which covers `[("billing", "Payments"), ("sales", "Pricing")]`. It does not
/// cover a *mixed* list where some options are described and others are null,
/// because the description type has to unify across the whole array. That is
/// what this builder is for:
///
/// ```
/// use typesafe::{ChoiceCriteria, Entry, QuestionSet};
///
/// let mut q = QuestionSet::new();
/// let key = q.choice(
///     "department",
///     "Which team should handle this?",
///     ChoiceCriteria::new()
///         .with("billing", "Payments, invoicing, refunds")
///         .with("other", Entry::null()),
/// );
/// # let _ = key;
/// ```
///
/// [`QuestionSet::choice`]: crate::QuestionSet::choice
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChoiceCriteria {
    entries: Vec<(String, Entry)>,
}

impl ChoiceCriteria {
    /// An empty set of options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an option, consuming and returning `self` so calls can chain.
    #[must_use]
    pub fn with<L, D>(mut self, label: L, description: D) -> Self
    where
        L: Into<String>,
        D: IntoEntry,
    {
        self.push(label, description);
        self
    }

    /// Add an option in place.
    pub fn push<L, D>(&mut self, label: L, description: D)
    where
        L: Into<String>,
        D: IntoEntry,
    {
        self.entries.push((label.into(), description.into_entry()));
    }

    /// Number of options.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no options have been added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl IntoIterator for ChoiceCriteria {
    type Item = (String, Entry);
    type IntoIter = std::vec::IntoIter<(String, Entry)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<L, D> Extend<(L, D)> for ChoiceCriteria
where
    L: Into<String>,
    D: IntoEntry,
{
    fn extend<I: IntoIterator<Item = (L, D)>>(&mut self, iter: I) {
        for (label, description) in iter {
            self.push(label, description);
        }
    }
}

impl<L, D> FromIterator<(L, D)> for ChoiceCriteria
where
    L: Into<String>,
    D: IntoEntry,
{
    fn from_iter<I: IntoIterator<Item = (L, D)>>(iter: I) -> Self {
        let mut criteria = ChoiceCriteria::new();
        criteria.extend(iter);
        criteria
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strings_become_json_strings() {
        assert_eq!(*"hi".into_entry().as_value(), json!("hi"));
        assert_eq!(*String::from("hi").into_entry().as_value(), json!("hi"));
    }

    #[test]
    fn unit_and_none_become_null() {
        assert!(().into_entry().is_null());
        assert!(None::<&str>.into_entry().is_null());
        assert!(Option::<Entry>::None.into_entry().is_null());
    }

    #[test]
    fn some_unwraps_to_the_inner_entry() {
        assert_eq!(*Some("hi").into_entry().as_value(), json!("hi"));
    }

    #[test]
    fn structured_criteria_survive_the_round_trip() {
        let tree = json!({"Cycling": {"Bike Bottles & Cages": {}}});
        assert_eq!(*tree.clone().into_entry().as_value(), tree);
    }

    #[test]
    fn entry_json_accepts_serialize_types() {
        #[derive(serde::Serialize)]
        struct Ticket<'a> {
            message: &'a str,
        }
        let entry = Entry::json(&Ticket { message: "help" }).unwrap();
        assert_eq!(*entry.as_value(), json!({"message": "help"}));
    }

    #[test]
    fn choice_criteria_preserves_insertion_order() {
        let criteria = ChoiceCriteria::new()
            .with("billing", "Payments")
            .with("technical", Entry::null())
            .with("sales", "Pricing");
        let labels: Vec<String> = criteria.into_iter().map(|(label, _)| label).collect();
        assert_eq!(labels, ["billing", "technical", "sales"]);
    }
}
