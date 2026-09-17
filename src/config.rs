//! Configuration: environment variables, defaults, and model names.

use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

/// Environment variable holding the API key.
pub const ENV_API_KEY: &str = "TYPESAFE_API_KEY";
/// Environment variable overriding the API root.
pub const ENV_BASE_URL: &str = "TYPESAFE_BASE_URL";
/// Environment variable overriding the default model.
pub const ENV_DEFAULT_MODEL: &str = "TYPESAFE_DEFAULT_MODEL";

/// The API root used when nothing overrides it.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// Per-attempt request timeout used when nothing overrides it.
///
/// Matches the other TypeSafe SDKs. Note this is per *attempt*: retries do not
/// share a total budget, so a request that retries twice can take up to three
/// times this long plus backoff.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// A model name.
///
/// Either an alias that moves as TypeSafe ships releases ([`Model::JEV_LATEST`],
/// [`Model::JEV_PREVIEW`]) or a pinned version like `jev-1.13.0`. Pin the
/// version if you have tuned confidence thresholds against one, since an alias
/// can change the answers under you.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model(Cow<'static, str>);

impl Model {
    /// `jev-latest` — the most recent stable release. The default.
    pub const JEV_LATEST: Model = Model(Cow::Borrowed("jev-latest"));

    /// `jev-preview` — the most recent release, official or not.
    pub const JEV_PREVIEW: Model = Model(Cow::Borrowed("jev-preview"));

    /// Name a model explicitly, e.g. a pinned `jev-1.13.0`.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Model(name.into())
    }

    /// The name as sent in the request body.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for Model {
    fn default() -> Self {
        Model::JEV_LATEST
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Model {
    fn from(name: String) -> Self {
        Model(Cow::Owned(name))
    }
}

impl From<&'static str> for Model {
    fn from(name: &'static str) -> Self {
        Model(Cow::Borrowed(name))
    }
}

/// Read an environment variable, treating blank values as unset.
///
/// Matches the JavaScript SDK, which documents that "empty or whitespace-only
/// environment values are ignored" — otherwise an exported-but-empty
/// `TYPESAFE_API_KEY` would shadow a key passed in code.
pub(crate) fn env_var(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Strip trailing slashes so joining a path cannot produce a double slash.
pub(crate) fn normalize_base_url(url: &str) -> String {
    url.trim_end_matches('/').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_the_stable_alias() {
        assert_eq!(Model::default().as_str(), "jev-latest");
    }

    #[test]
    fn models_can_be_pinned() {
        assert_eq!(Model::new("jev-1.13.0").as_str(), "jev-1.13.0");
        assert_eq!(Model::from("jev-1.13.0").to_string(), "jev-1.13.0");
    }

    #[test]
    fn trailing_slashes_are_stripped() {
        assert_eq!(
            normalize_base_url("https://api.typesafe.ai///"),
            "https://api.typesafe.ai"
        );
        assert_eq!(
            normalize_base_url("https://api.typesafe.ai"),
            "https://api.typesafe.ai"
        );
    }

    #[test]
    fn blank_environment_values_are_ignored() {
        // SAFETY: single-threaded test, and the variable is unique to it.
        let name = "TYPESAFE_TEST_BLANK_VAR";
        unsafe { std::env::set_var(name, "   ") };
        assert_eq!(env_var(name), None);
        unsafe { std::env::set_var(name, "  value  ") };
        assert_eq!(env_var(name).as_deref(), Some("value"));
        unsafe { std::env::remove_var(name) };
        assert_eq!(env_var(name), None);
    }
}
