//! The models available to an account.

use serde::Deserialize;

/// Metadata for one model your account can send in the `model` field.
///
/// The listing covers the aliases (`jev-latest`, `jev-preview`). Pinned
/// version ids like `jev-1.13.0` are accepted whether or not they appear here.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct ModelCard {
    /// The name to send in the `model` field.
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Release date, as the API formats it.
    #[serde(default)]
    pub release_date: String,
}

/// The `GET /v1/models` response body.
///
/// The endpoint wraps the list in an object; callers get the `Vec` directly.
#[derive(Debug, Deserialize)]
pub(crate) struct WireModels {
    #[serde(default)]
    pub(crate) models: Vec<ModelCard>,
}
