//! Wire format and transport plumbing shared by the async and blocking clients.
//!
//! Everything here is runtime-agnostic on purpose: building the request body,
//! classifying a response, and deciding whether to retry are pure functions,
//! so the two client loops differ only in how they sleep and how they await.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde_json::Value;

use crate::config::Model;
use crate::error::{ApiError, Error};
use crate::question::{NoulCriteria, Question, QuestionSet};
use crate::retry::{RetryPolicy, retry_after};

/// Path of the evaluation endpoint.
pub(crate) const SYSTEM_ONE_PATH: &str = "/v1/systemone";
/// Path of the model listing endpoint.
pub(crate) const MODELS_PATH: &str = "/v1/models";

/// Header carrying the id to quote when reporting a problem.
pub(crate) const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";

/// This crate's `User-Agent`.
pub(crate) fn user_agent() -> String {
    format!("typesafe-ai-rs/{}", env!("CARGO_PKG_VERSION"))
}

/// The `POST /v1/systemone` request body.
///
/// Serialized through a private type so the JSON shape is not part of the
/// public API, and so question names keep their insertion order.
struct RequestBody<'a> {
    state: &'a Value,
    model: &'a Model,
    questions: &'a QuestionSet,
}

impl Serialize for RequestBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("state", self.state)?;
        map.serialize_entry("model", self.model.as_str())?;
        map.serialize_entry("questions", &QuestionsBody(self.questions))?;
        map.end()
    }
}

struct QuestionsBody<'a>(&'a QuestionSet);

impl Serialize for QuestionsBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, question) in self.0.questions() {
            map.serialize_entry(name, &QuestionBody(question))?;
        }
        map.end()
    }
}

struct QuestionBody<'a>(&'a Question);

impl Serialize for QuestionBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", self.0.tag())?;

        match self.0 {
            Question::Noul {
                instructions,
                criteria,
            } => {
                map.serialize_entry("instructions", instructions.as_value())?;
                if let Some(criteria) = criteria {
                    map.serialize_entry("criteria", &NoulCriteriaBody(criteria))?;
                }
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                map.serialize_entry("instructions", instructions.as_value())?;
                map.serialize_entry("criteria", &ChoiceCriteriaBody(criteria))?;
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                map.serialize_entry("instructions", instructions.as_value())?;
                map.serialize_entry("criteria", &ScoreCriteriaBody(criteria))?;
            }
        }

        map.end()
    }
}

struct NoulCriteriaBody<'a>(&'a NoulCriteria);

impl Serialize for NoulCriteriaBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("true", self.0.yes.as_value())?;
        map.serialize_entry("false", self.0.no.as_value())?;
        map.end()
    }
}

struct ChoiceCriteriaBody<'a>(&'a [(String, crate::entry::Entry)]);

impl Serialize for ChoiceCriteriaBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // A map rather than `collect_map`, so option order survives: the
        // request is JSON, and serde_json writes entries in the order given.
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (label, description) in self.0 {
            map.serialize_entry(label, description.as_value())?;
        }
        map.end()
    }
}

struct ScoreCriteriaBody<'a>(&'a [crate::entry::Entry]);

impl Serialize for ScoreCriteriaBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for level in self.0 {
            seq.serialize_element(level.as_value())?;
        }
        seq.end()
    }
}

/// Render the evaluation request body.
pub(crate) fn system_one_body(
    questions: &QuestionSet,
    state: &Value,
    model: &Model,
) -> Result<Vec<u8>, Error> {
    serde_json::to_vec(&RequestBody {
        state,
        model,
        questions,
    })
    .map_err(|source| Error::Decode {
        source,
        request_id: None,
    })
}

/// Join the base URL and an endpoint path.
pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    format!("{base_url}{path}")
}

/// Pull the request id out of response headers.
pub(crate) fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Turn a non-2xx response into an [`ApiError`].
pub(crate) fn api_error(status: u16, headers: &HeaderMap, body: &[u8]) -> ApiError {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let message = parsed
        .as_ref()
        .and_then(error_message)
        .or_else(|| {
            let text = String::from_utf8_lossy(body).trim().to_owned();
            (!text.is_empty()).then_some(text)
        })
        .unwrap_or_else(|| default_message(status).to_owned());

    let error_type = parsed.as_ref().and_then(error_type);

    ApiError {
        status,
        message,
        error_type,
        body: parsed,
        request_id: request_id(headers),
        retry_after: retry_after(headers),
    }
}

/// Keys that have been observed to wrap the real error object.
const ERROR_WRAPPERS: [&str; 2] = ["detail", "error"];

fn as_message(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
        // Last resort: better to show the raw JSON than to drop the detail.
        Value::Object(_) | Value::Array(_) => Some(value.to_string()),
        _ => None,
    }
}

/// Dig a human-readable message out of an error body.
///
/// The live API nests it one level:
///
/// ```json
/// {"detail": {"error_type": "authentication_error", "message": "..."}}
/// ```
///
/// so the wrappers are unwrapped first. The flat spellings are still tried
/// afterwards, and the raw text is the final fallback, because the docs
/// specify nothing beyond "a JSON body describing what went wrong" — the shape
/// can change without warning.
fn error_message(body: &Value) -> Option<String> {
    for key in ERROR_WRAPPERS {
        if let Some(message) = body.get(key).and_then(|nested| nested.get("message")).and_then(as_message) {
            return Some(message);
        }
    }

    for key in ["message", "detail", "error", "error_description"] {
        if let Some(value) = body.get(key)
            && let Some(message) = as_message(value)
        {
            return Some(message);
        }
    }

    None
}

/// The machine-readable error class, e.g. `authentication_error`.
///
/// Not documented, but the API returns it consistently and it carries
/// information the status code does not — a missing API key comes back as
/// `403` with an `authentication_error`.
fn error_type(body: &Value) -> Option<String> {
    for key in ERROR_WRAPPERS {
        if let Some(kind) = body
            .get(key)
            .and_then(|nested| nested.get("error_type"))
            .and_then(Value::as_str)
        {
            return Some(kind.to_owned());
        }
    }

    body.get("error_type")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn default_message(status: u16) -> &'static str {
    match status {
        400 => "bad request",
        401 => "missing or invalid API key",
        403 => "permission denied",
        404 => "not found",
        422 => "the request body failed validation",
        429 => "rate limit exceeded",
        529 => "TypeSafe is temporarily overloaded",
        500..=599 => "server error",
        _ => "unexpected response",
    }
}

/// Map a transport failure onto the crate's error type.
pub(crate) fn transport_error(error: reqwest::Error, attempts: u32) -> Error {
    if error.is_timeout() {
        Error::Timeout {
            attempts,
            source: error,
        }
    } else if error.is_connect() {
        Error::Connect {
            attempts,
            source: error,
        }
    } else {
        Error::Transport(error)
    }
}

/// Tracks how many retries a single call has spent.
///
/// Shared by both clients so the two loops cannot drift apart in when they
/// give up; all that differs is how each one waits.
pub(crate) struct Retries<'a> {
    policy: &'a RetryPolicy,
    spent: u32,
}

impl<'a> Retries<'a> {
    pub(crate) fn new(policy: &'a RetryPolicy) -> Self {
        Retries { policy, spent: 0 }
    }

    /// Attempts made so far, counting the first.
    pub(crate) fn attempts(&self) -> u32 {
        self.spent + 1
    }

    /// How long to wait before retrying this status, or `None` to give up.
    pub(crate) fn after_status(&mut self, status: u16, headers: &HeaderMap) -> Option<Duration> {
        if self.spent >= self.policy.max_retries || !self.policy.retries_status(status) {
            return None;
        }
        let delay = self.policy.delay(self.spent, retry_after(headers));
        self.spent += 1;
        Some(delay)
    }

    /// How long to wait before retrying this transport failure, or `None`.
    pub(crate) fn after_transport(&mut self, error: &reqwest::Error) -> Option<Duration> {
        if self.spent >= self.policy.max_retries {
            return None;
        }

        let retryable = (error.is_timeout() && self.policy.retry_timeouts)
            // An interrupted response body is a connection failure too, which
            // is how the official SDKs classify it.
            || ((error.is_connect() || error.is_body()) && self.policy.retry_connection_errors);

        if !retryable {
            return None;
        }

        let delay = self.policy.delay(self.spent, None);
        self.spent += 1;
        Some(delay)
    }
}

/// Build the header map every request starts from.
pub(crate) fn base_headers(
    api_key: &str,
    extra: &HeaderMap,
) -> Result<HeaderMap, crate::error::ConfigError> {
    let mut headers = HeaderMap::new();

    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        reqwest::header::USER_AGENT,
        HeaderValue::from_str(&user_agent()).expect("the user agent is always valid"),
    );

    for (name, value) in extra {
        headers.insert(name.clone(), value.clone());
    }

    let mut authorization = HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|err| {
        crate::error::ConfigError::InvalidHeader {
            name: "authorization".to_owned(),
            reason: err.to_string(),
        }
    })?;
    // Keeps the key out of `{:?}` output on the header map.
    authorization.set_sensitive(true);
    headers.insert(reqwest::header::AUTHORIZATION, authorization);

    Ok(headers)
}

/// Convert caller-supplied header pairs, reporting which one was bad.
pub(crate) fn header_pair(
    name: &str,
    value: &str,
) -> Result<(HeaderName, HeaderValue), crate::error::ConfigError> {
    let header_name =
        HeaderName::try_from(name).map_err(|err| crate::error::ConfigError::InvalidHeader {
            name: name.to_owned(),
            reason: err.to_string(),
        })?;
    let header_value =
        HeaderValue::from_str(value).map_err(|err| crate::error::ConfigError::InvalidHeader {
            name: name.to_owned(),
            reason: err.to_string(),
        })?;
    Ok((header_name, header_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::Entry;
    use serde_json::json;

    fn body_of(questions: &QuestionSet, state: Value) -> Value {
        let bytes = system_one_body(questions, &state, &Model::JEV_LATEST).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn the_request_matches_the_documented_example() {
        // Straight from https://docs.typesafe.ai/api
        let mut q = QuestionSet::new();
        q.noul_with(
            "is_urgent",
            "Does this convey urgency?",
            "Explicitly time-sensitive",
            "No urgency expressed",
        );

        assert_eq!(
            body_of(&q, json!("Help! My payouts have been failing for 3 days.")),
            json!({
                "state": "Help! My payouts have been failing for 3 days.",
                "model": "jev-latest",
                "questions": {
                    "is_urgent": {
                        "type": "noul",
                        "instructions": "Does this convey urgency?",
                        "criteria": {
                            "true": "Explicitly time-sensitive",
                            "false": "No urgency expressed"
                        }
                    }
                }
            })
        );
    }

    #[test]
    fn a_bare_noul_sends_no_criteria() {
        let mut q = QuestionSet::new();
        q.noul("is_urgent", "Does this convey urgency?");
        let body = body_of(&q, json!("hi"));
        assert!(body["questions"]["is_urgent"].get("criteria").is_none());
    }

    #[test]
    fn choice_and_score_match_the_documented_examples() {
        let mut q = QuestionSet::new();
        q.choice("department", "Which team should handle this?", [
            ("billing", "Payments, invoicing, refunds"),
            ("technical", "Bugs, outages, integrations"),
            ("sales", "Pricing, upgrades, new accounts"),
        ]);
        q.score("frustration", "How frustrated is the customer?", [
            "Calm",
            "Frustrated",
            "Very angry",
        ]);

        let body = body_of(&q, json!("Help!"));
        assert_eq!(
            body["questions"]["department"],
            json!({
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"
                }
            })
        );
        assert_eq!(
            body["questions"]["frustration"],
            json!({
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"]
            })
        );
    }

    #[test]
    fn choice_option_order_is_preserved() {
        let mut q = QuestionSet::new();
        q.choice_labels("taxonomy", "Which department?", ["zeta", "alpha", "middle"]);
        let bytes = system_one_body(&q, &json!(null), &Model::JEV_LATEST).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let zeta = text.find("zeta").unwrap();
        let alpha = text.find("alpha").unwrap();
        let middle = text.find("middle").unwrap();
        assert!(zeta < alpha && alpha < middle, "{text}");
    }

    #[test]
    fn structured_state_and_criteria_round_trip() {
        let mut q = QuestionSet::new();
        q.choice("dept", json!({"task": "classify"}), [(
            "sporting",
            json!({"Cycling": {"Bike Bottles & Cages": {}}}),
        )]);

        let body = body_of(&q, json!({"listing": "bottle", "price": 12}));
        assert_eq!(body["state"], json!({"listing": "bottle", "price": 12}));
        assert_eq!(
            body["questions"]["dept"]["criteria"]["sporting"],
            json!({"Cycling": {"Bike Bottles & Cages": {}}})
        );
        assert_eq!(
            body["questions"]["dept"]["instructions"],
            json!({"task": "classify"})
        );
    }

    #[test]
    fn undescribed_choice_options_serialize_as_null() {
        let mut q = QuestionSet::new();
        q.choice("dept", "Which?", [("billing", Entry::null())]);
        let body = body_of(&q, json!(null));
        assert_eq!(body["questions"]["dept"]["criteria"]["billing"], json!(null));
    }

    #[test]
    fn null_state_is_sent_explicitly() {
        let mut q = QuestionSet::new();
        q.noul("q", "?");
        assert_eq!(body_of(&q, json!(null))["state"], json!(null));
    }

    #[test]
    fn urls_join_without_doubling_slashes() {
        assert_eq!(
            endpoint("https://api.typesafe.ai", SYSTEM_ONE_PATH),
            "https://api.typesafe.ai/v1/systemone"
        );
    }

    #[test]
    fn error_messages_are_dug_out_of_the_usual_shapes() {
        let headers = HeaderMap::new();
        let from = |body: Value| api_error(422, &headers, body.to_string().as_bytes()).message;

        assert_eq!(from(json!({"message": "bad field"})), "bad field");
        assert_eq!(from(json!({"detail": "bad field"})), "bad field");
        assert_eq!(from(json!({"error": {"message": "bad field"}})), "bad field");
        assert_eq!(from(json!({"error": "bad field"})), "bad field");
    }

    #[test]
    fn the_live_error_shape_yields_a_clean_message() {
        // Captured verbatim from api.typesafe.ai.
        let body = br#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}"#;
        let error = api_error(401, &HeaderMap::new(), body);

        assert_eq!(
            error.message,
            "Cannot authenticate with the server. Please check your API key and try again."
        );
        assert_eq!(error.error_type.as_deref(), Some("authentication_error"));
        assert!(error.is_auth());
        // The whole JSON object must not leak into the message.
        assert!(!error.message.contains("error_type"), "{}", error.message);
    }

    #[test]
    fn a_missing_api_key_is_an_auth_error_despite_the_403() {
        // The API answers a missing Authorization header with 403, not 401.
        let body = br#"{"detail":{"error_type":"authentication_error","message":"Must supply an API key! Check your request and try again."}}"#;
        let error = api_error(403, &HeaderMap::new(), body);

        assert!(error.is_auth(), "a missing key should read as an auth failure");
        assert!(!error.is_permission_denied());
    }

    #[test]
    fn a_genuine_403_is_still_permission_denied() {
        let error = api_error(403, &HeaderMap::new(), br#"{"detail":{"message":"no access"}}"#);
        assert!(error.is_permission_denied());
        assert!(!error.is_auth());
    }

    #[test]
    fn an_unknown_model_is_a_bad_request() {
        let body = br#"{"detail":{"error_type":"api_usage_error","message":"Unknown model: gpt-9"}}"#;
        let error = api_error(400, &HeaderMap::new(), body);
        assert!(error.is_bad_request());
        assert_eq!(error.message, "Unknown model: gpt-9");
        assert_eq!(error.error_type.as_deref(), Some("api_usage_error"));
    }

    #[test]
    fn an_unparseable_error_body_falls_back_to_text_then_status() {
        let headers = HeaderMap::new();
        assert_eq!(api_error(500, &headers, b"upstream exploded").message, "upstream exploded");
        assert_eq!(api_error(429, &headers, b"").message, "rate limit exceeded");
        assert_eq!(api_error(529, &headers, b"").message, "TypeSafe is temporarily overloaded");
    }

    #[test]
    fn the_api_key_is_marked_sensitive() {
        let headers = base_headers("sk-secret", &HeaderMap::new()).unwrap();
        assert!(headers[reqwest::header::AUTHORIZATION].is_sensitive());
        assert!(!format!("{headers:?}").contains("sk-secret"));
    }

    #[test]
    fn default_headers_cannot_clobber_the_api_key() {
        let mut extra = HeaderMap::new();
        extra.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer nope"),
        );
        let headers = base_headers("sk-real", &extra).unwrap();
        assert_eq!(headers[reqwest::header::AUTHORIZATION], "Bearer sk-real");
    }

    #[test]
    fn retries_stop_at_the_policy_limit() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        let headers = HeaderMap::new();
        let mut retries = Retries::new(&policy);

        assert_eq!(retries.attempts(), 1);
        assert!(retries.after_status(429, &headers).is_some());
        assert!(retries.after_status(429, &headers).is_some());
        assert!(retries.after_status(429, &headers).is_none());
        assert_eq!(retries.attempts(), 3);
    }

    #[test]
    fn non_retryable_statuses_are_not_retried() {
        let policy = RetryPolicy::default();
        let mut retries = Retries::new(&policy);
        assert!(retries.after_status(422, &HeaderMap::new()).is_none());
        assert!(retries.after_status(401, &HeaderMap::new()).is_none());
    }
}
