//! A Rust client for the [TypeSafe AI](https://docs.typesafe.ai) System One API.
//!
//! TypeSafe's Jev is a *System One* model: instead of generating prose, it
//! evaluates some **state** against a set of typed **questions** and hands back
//! structured answers with calibrated probabilities — values your code can
//! branch on directly.
//!
//! ```no_run
//! use typesafe::{Client, QuestionSet};
//!
//! # async fn demo() -> Result<(), typesafe::Error> {
//! let client = Client::new()?; // reads TYPESAFE_API_KEY
//!
//! let mut q = QuestionSet::new();
//! let urgent = q.noul("is_urgent", "Does this convey urgency?");
//! let dept = q.choice("department", "Which team should handle this?", [
//!     ("billing", "Payments, invoicing, refunds"),
//!     ("technical", "Bugs, outages, integrations"),
//!     ("sales", "Pricing, upgrades, new accounts"),
//! ]);
//! let frustration = q.score("frustration", "How frustrated is the customer?", [
//!     "Calm",
//!     "Frustrated",
//!     "Very angry",
//! ]);
//!
//! let answers = client
//!     .system_one(&q)
//!     .state("Help! My payouts have been failing for 3 days.")
//!     .send()
//!     .await?;
//!
//! if answers[urgent].noul > 0.8 && answers[dept].confidence > 0.7 {
//!     escalate(&answers[dept].choice, answers[frustration].score);
//! }
//! # fn escalate(_team: &str, _score: f64) {}
//! # Ok(())
//! # }
//! ```
//!
//! # Typed answers without stringly-typed lookups
//!
//! Each question builder returns a [`Key`] carrying that question's type, and
//! the response is indexed by those keys. `answers[urgent]` is a
//! [`NoulAnswer`], `answers[dept]` is a [`ChoiceAnswer`] — no `match`, no
//! `unwrap`, no chance of asking a Choice question and reading a Score.
//!
//! Indexing cannot fail for a key from the set that produced the response: the
//! response is checked on arrival, so a missing or mistyped answer surfaces as
//! [`Error::Protocol`] at the call site. A key from a *different*
//! [`QuestionSet`] panics rather than quietly return a plausible number from
//! the wrong question; [`SystemOne::get`] is the non-panicking form.
//!
//! # Ask everything at once
//!
//! Every question in a set is evaluated in one call, in parallel. TypeSafe's
//! docs put it plainly: "asking a question you might not need is close to
//! free." Build one [`QuestionSet`], keep it, and reuse it — keys stay valid
//! for its lifetime, so it belongs in a struct next to the client.
//!
//! # Runtime support, and a caveat
//!
//! [`Client`] needs a **tokio** runtime, because `reqwest` does. This is not a
//! choice this crate makes and not one it can forward: every async path in
//! reqwest is wired to tokio, and it exposes no `async-std`, `smol`, or
//! `runtime-*` feature to pass along.
//!
//! What *is* forwarded are the transport knobs reqwest actually has:
//!
//! | Feature | Effect |
//! |---|---|
//! | `rustls-tls` *(default)* | TLS via rustls |
//! | `native-tls` | TLS via the platform library |
//! | `gzip`, `brotli`, `deflate`, `zstd` | response decompression |
//! | `socks` | SOCKS5 proxies |
//! | `http2` *(default)*, `http3` | HTTP versions |
//! | `blocking` | the synchronous [`blocking::Client`] |
//!
//! On a non-tokio runtime, either enable `blocking` (which runs its own tokio
//! reactor on a background thread) or wrap calls in
//! [`async_compat::Compat`](https://docs.rs/async-compat).
//!
//! `http3` additionally needs `RUSTFLAGS="--cfg reqwest_unstable"`, which is
//! reqwest's requirement — so `--all-features` does not build.
//!
//! # Errors and retries
//!
//! [`RetryPolicy`] defaults to what the official SDKs do: two retries with
//! exponential backoff from 500ms to a 5s ceiling, 25% jitter, retrying `408`,
//! `429` and `5xx` plus connection errors and timeouts, honouring
//! `Retry-After`. TypeSafe warns that rate limits move without notice while
//! they scale, so leaving this on is worthwhile.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub(crate) mod sealed {
    /// Prevents downstream crates from implementing traits that are part of
    /// this crate's type-level bookkeeping.
    pub trait Sealed {}
}

mod client;
mod config;
mod entry;
mod error;
mod http;
mod models;
mod question;
mod response;
mod retry;

#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub mod blocking;

pub use client::{Client, ClientBuilder, ModelsRequest, SystemOneRequest};
pub use config::{
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, ENV_API_KEY, ENV_BASE_URL, ENV_DEFAULT_MODEL, Model,
};
pub use entry::{ChoiceCriteria, Entry, IntoEntry};
pub use error::{ApiError, ConfigError, Error, KeyError, ProtocolError, QuestionSetError};
pub use models::ModelCard;
pub use question::{
    Choice, ChoiceKey, Key, Noul, NoulKey, QuestionKind, QuestionSet, Score, ScoreKey,
};
pub use response::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOne, Usage};
pub use retry::RetryPolicy;

/// This crate's version, as reported in the `User-Agent`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
