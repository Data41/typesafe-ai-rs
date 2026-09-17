//! Error types.
//!
//! Everything fallible in this crate returns [`Error`]. Its variants separate
//! the four failures you handle differently: bad configuration, a question set
//! you built wrong, something going wrong on the wire, and the server
//! answering in a shape that contradicts what was asked.

use std::time::Duration;

use serde_json::Value;

/// The error type for every fallible operation in this crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The client could not be configured — most often a missing API key.
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// The question set was rejected before any request was sent.
    #[error(transparent)]
    QuestionSet(#[from] QuestionSetError),

    /// The request timed out. Reported after retries are exhausted.
    #[error("request timed out after {attempts} attempt(s)")]
    Timeout {
        /// Total attempts made, including the first.
        attempts: u32,
        /// The underlying transport error.
        #[source]
        source: reqwest::Error,
    },

    /// The request never reached the API (DNS, TLS, connection refused).
    /// Reported after retries are exhausted.
    #[error("could not reach the TypeSafe API after {attempts} attempt(s)")]
    Connect {
        /// Total attempts made, including the first.
        attempts: u32,
        /// The underlying transport error.
        #[source]
        source: reqwest::Error,
    },

    /// Any other transport-layer failure.
    #[error("HTTP transport error")]
    Transport(#[source] reqwest::Error),

    /// The API returned a non-2xx response.
    ///
    /// Boxed because it is by far the largest variant, and every `Result` in
    /// the crate would otherwise pay for it.
    #[error(transparent)]
    Api(Box<ApiError>),

    /// The response body was not the JSON we expected.
    #[error("could not decode the API response{}", OptionalRequestId(.request_id))]
    Decode {
        /// The underlying parse failure.
        #[source]
        source: serde_json::Error,
        /// Value of the `x-typesafe-request-id` response header, when present.
        request_id: Option<String>,
    },

    /// The response was valid JSON but contradicted the request — a question
    /// went unanswered, or came back as the wrong type.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

impl From<ApiError> for Error {
    fn from(error: ApiError) -> Self {
        Error::Api(Box::new(error))
    }
}

impl Error {
    /// The `x-typesafe-request-id` header, when the failure carried one.
    ///
    /// Worth logging alongside any failure you report to TypeSafe support.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Error::Api(err) => err.request_id.as_deref(),
            Error::Decode { request_id, .. } => request_id.as_deref(),
            _ => None,
        }
    }

    /// The API error, when this failure came from a non-2xx response.
    pub fn as_api(&self) -> Option<&ApiError> {
        match self {
            Error::Api(err) => Some(err),
            _ => None,
        }
    }

    /// True when the failure is one a later identical request might survive:
    /// a timeout, a connection failure, or a retryable status that outlived
    /// the retry policy.
    pub fn is_transient(&self) -> bool {
        match self {
            Error::Timeout { .. } | Error::Connect { .. } => true,
            Error::Api(err) => err.is_rate_limit() || err.is_overloaded() || err.is_server_error(),
            _ => false,
        }
    }
}

/// A non-2xx response from the API.
///
/// Rather than one variant per status code — the set is open-ended and would
/// churn — this carries the raw status with predicates for the codes the
/// [API reference](https://docs.typesafe.ai/api) documents.
#[derive(Debug, Clone, thiserror::Error)]
#[error("TypeSafe API error {status}: {message}{}", OptionalRequestId(.request_id))]
#[non_exhaustive]
pub struct ApiError {
    /// HTTP status code.
    pub status: u16,
    /// Human-readable message, pulled from the response body when it has one.
    pub message: String,
    /// The machine-readable error class, e.g. `authentication_error` or
    /// `api_usage_error`.
    ///
    /// Undocumented, but returned consistently, and it carries information the
    /// status code does not — see [`is_auth`](Self::is_auth).
    pub error_type: Option<String>,
    /// The parsed response body, when it was JSON.
    pub body: Option<Value>,
    /// Value of the `x-typesafe-request-id` response header, when present.
    pub request_id: Option<String>,
    /// Server-requested delay from `Retry-After` / `retry-after-ms`.
    pub retry_after: Option<Duration>,
}

impl ApiError {
    /// `400 Bad Request`.
    pub fn is_bad_request(&self) -> bool {
        self.status == 400
    }

    /// The API key was missing, invalid, or rejected.
    ///
    /// Covers more than `401`: the API answers a *missing* `Authorization`
    /// header with `403` and an `authentication_error`, so keying off the
    /// status alone would miss the most common misconfiguration there is.
    pub fn is_auth(&self) -> bool {
        self.status == 401
            || (self.status == 403 && self.error_type.as_deref() == Some("authentication_error"))
    }

    /// `403 Forbidden` for a reason other than authentication.
    pub fn is_permission_denied(&self) -> bool {
        self.status == 403 && !self.is_auth()
    }

    /// `404 Not Found`.
    pub fn is_not_found(&self) -> bool {
        self.status == 404
    }

    /// `422 Unprocessable Entity` — the request body failed validation.
    pub fn is_unprocessable(&self) -> bool {
        self.status == 422
    }

    /// `429 Too Many Requests` — rate limited.
    pub fn is_rate_limit(&self) -> bool {
        self.status == 429
    }

    /// `529 Overloaded` — TypeSafe is temporarily overloaded.
    pub fn is_overloaded(&self) -> bool {
        self.status == 529
    }

    /// Any `5xx`.
    pub fn is_server_error(&self) -> bool {
        (500..600).contains(&self.status)
    }

    /// The delay the server asked for, if it asked for one.
    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }
}

/// The client could not be configured.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// No API key was supplied and `TYPESAFE_API_KEY` was unset or blank.
    #[error(
        "no API key: pass one to ClientBuilder::api_key, or set the TYPESAFE_API_KEY \
         environment variable"
    )]
    MissingApiKey,

    /// The base URL could not be used to build request URLs.
    #[error("invalid base URL {url:?}: {reason}")]
    InvalidBaseUrl {
        /// The offending URL.
        url: String,
        /// Why it was rejected.
        reason: String,
    },

    /// A default header name or value was not valid HTTP.
    #[error("invalid default header {name:?}: {reason}")]
    InvalidHeader {
        /// The offending header name.
        name: String,
        /// Why it was rejected.
        reason: String,
    },

    /// The underlying `reqwest` client could not be built.
    #[error("could not build the HTTP client")]
    Http(#[source] std::sync::Arc<reqwest::Error>),
}

/// A [`QuestionSet`](crate::QuestionSet) that cannot be sent as-is.
///
/// Question builders are infallible so they can hand back a key. Problems are
/// recorded on the set and surfaced here, by
/// [`QuestionSet::validate`](crate::QuestionSet::validate) or by `send()`
/// before any request goes out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum QuestionSetError {
    /// The set has no questions. The API requires at least one.
    #[error("the question set is empty; add at least one question")]
    Empty,

    /// Two questions share a name. Their answers would collide, because the
    /// API keys answers by the names you choose.
    #[error("duplicate question name {name:?}")]
    DuplicateName {
        /// The repeated name.
        name: String,
    },

    /// A Choice question has no options. The API requires at least one.
    #[error("choice question {name:?} has no options")]
    EmptyChoiceCriteria {
        /// The offending question's name.
        name: String,
    },

    /// A Score question has fewer than the two levels the API requires.
    ///
    /// Only reachable through
    /// [`score_iter`](crate::QuestionSet::score_iter); the array-taking
    /// [`score`](crate::QuestionSet::score) rejects this at compile time.
    #[error("score question {name:?} needs at least 2 levels, got {len}")]
    TooFewScoreCriteria {
        /// The offending question's name.
        name: String,
        /// How many levels were supplied.
        len: usize,
    },
}

/// The response contradicted the request.
///
/// These are the invariants that make indexing a response with a typed key
/// infallible, so a violation is an error at the call site rather than a panic
/// somewhere downstream.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProtocolError {
    /// A question we asked came back unanswered.
    #[error("the API returned no answer for question {name:?}")]
    MissingAnswer {
        /// The unanswered question's name.
        name: String,
    },

    /// An answer came back as a different type than the question asked for.
    #[error("question {name:?} is a {expected} question but the API answered with {found}")]
    WrongAnswerType {
        /// The question's name.
        name: String,
        /// The type we asked for.
        expected: &'static str,
        /// The type we got.
        found: String,
    },

    /// A Score answer's `legend` or `probabilities` were not keyed by a
    /// contiguous run of level indices starting at zero.
    #[error("score answer for {name:?} has malformed level keys: {reason}")]
    MalformedScoreKeys {
        /// The question's name.
        name: String,
        /// What was wrong with them.
        reason: String,
    },
}

/// A typed key could not be resolved against a response.
///
/// Returned by [`SystemOne::get`](crate::SystemOne::get). Indexing with `[]`
/// panics on these instead.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum KeyError {
    /// The key came from a different [`QuestionSet`](crate::QuestionSet) than
    /// the one that produced this response.
    ///
    /// Silently resolving it would hand back a plausible-looking number from
    /// the wrong question, so it is refused.
    #[error(
        "this key belongs to a different QuestionSet (key set #{key_set}, response set #{response_set})"
    )]
    ForeignQuestionSet {
        /// Identifier of the set the key came from.
        key_set: u64,
        /// Identifier of the set that produced the response.
        response_set: u64,
    },

    /// The key's position is past the end of the response.
    #[error("question index {index} is out of range for a response with {len} answer(s)")]
    OutOfRange {
        /// The key's position.
        index: u32,
        /// How many answers the response holds.
        len: usize,
    },

    /// The answer at that position is not the type the key expects.
    #[error("expected a {expected} answer, found {found}")]
    WrongType {
        /// The type the key expects.
        expected: &'static str,
        /// The type actually stored.
        found: &'static str,
    },
}

/// Renders ` (request id: …)` when there is one, and nothing when there isn't.
struct OptionalRequestId<'a>(&'a Option<String>);

impl std::fmt::Display for OptionalRequestId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(id) => write!(f, " (request id: {id})"),
            None => Ok(()),
        }
    }
}
