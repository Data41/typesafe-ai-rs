//! The asynchronous client.

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::HeaderMap;
use reqwest::{Method, Url};
use serde::Serialize;
use serde_json::Value;

use crate::config::{
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, ENV_API_KEY, ENV_BASE_URL, ENV_DEFAULT_MODEL, Model, env_var,
    normalize_base_url,
};
use crate::error::{ConfigError, Error};
use crate::http;
use crate::models::{ModelCard, WireModels};
use crate::question::QuestionSet;
use crate::response::{SystemOne, WireResponse};
use crate::retry::RetryPolicy;

/// An asynchronous client for the TypeSafe AI API.
///
/// Cheap to clone — clones share one connection pool — so build it once and
/// pass clones around.
///
/// # Runtime
///
/// This client requires a **tokio** runtime, because `reqwest` requires one.
/// There is no runtime selection to forward: every async path in reqwest is
/// wired to tokio. On `async-std`, `smol`, or any other executor, either use
/// the [`blocking`](crate::blocking) client (behind the `blocking` feature) or
/// wrap calls in [`async_compat::Compat`](https://docs.rs/async-compat).
///
/// # Example
///
/// ```no_run
/// use typesafe::{Client, QuestionSet};
///
/// # async fn demo() -> Result<(), typesafe::Error> {
/// let client = Client::new()?; // reads TYPESAFE_API_KEY
///
/// let mut q = QuestionSet::new();
/// let urgent = q.noul("is_urgent", "Does this convey urgency?");
///
/// let answers = client
///     .system_one(&q)
///     .state("Help! My payouts have been failing for 3 days.")
///     .send()
///     .await?;
///
/// println!("urgency: {:.0}%", answers[urgent].noul * 100.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    default_model: Model,
    default_headers: HeaderMap,
    timeout: Duration,
    retry: RetryPolicy,
}

impl Client {
    /// Build a client from the environment.
    ///
    /// Reads `TYPESAFE_API_KEY`, and optionally `TYPESAFE_BASE_URL` and
    /// `TYPESAFE_DEFAULT_MODEL`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::MissingApiKey`] when `TYPESAFE_API_KEY` is unset or blank.
    pub fn new() -> Result<Self, Error> {
        Client::builder().build()
    }

    /// Start configuring a client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// The API root, with trailing slashes removed.
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    /// The model used when a request does not name one.
    pub fn default_model(&self) -> &Model {
        &self.inner.default_model
    }

    /// The per-attempt timeout.
    pub fn timeout(&self) -> Duration {
        self.inner.timeout
    }

    /// The retry policy applied when a call does not override it.
    pub fn retry(&self) -> &RetryPolicy {
        &self.inner.retry
    }

    /// Evaluate a state against a set of questions.
    ///
    /// Returns a builder; nothing is sent until you call
    /// [`send`](SystemOneRequest::send).
    pub fn system_one<'a>(&'a self, questions: &'a QuestionSet) -> SystemOneRequest<'a> {
        SystemOneRequest {
            client: self,
            questions,
            state: Ok(Value::Null),
            model: None,
            options: CallOptions::default(),
        }
    }

    /// List the models available to the account.
    ///
    /// Returns a builder; nothing is sent until you call
    /// [`send`](ModelsRequest::send).
    pub fn models(&self) -> ModelsRequest<'_> {
        ModelsRequest {
            client: self,
            options: CallOptions::default(),
        }
    }

    fn request_headers(&self, per_call: &HeaderMap) -> Result<HeaderMap, ConfigError> {
        let mut extra = self.inner.default_headers.clone();
        for (name, value) in per_call {
            extra.insert(name.clone(), value.clone());
        }
        http::base_headers(&self.inner.api_key, &extra)
    }

    /// Send a request, retrying per policy, and return the body bytes.
    async fn execute(
        &self,
        method: Method,
        url: &str,
        body: Option<Vec<u8>>,
        options: &CallOptions,
    ) -> Result<(Vec<u8>, Option<String>), Error> {
        if let Some(error) = &options.header_error {
            return Err(error.clone().into());
        }

        let policy = options.retry.as_ref().unwrap_or(&self.inner.retry);
        let timeout = options.timeout.unwrap_or(self.inner.timeout);
        let headers = self.request_headers(&options.headers)?;
        let mut retries = http::Retries::new(policy);

        loop {
            let mut request = self
                .inner
                .http
                .request(method.clone(), url)
                .headers(headers.clone())
                .timeout(timeout);

            if let Some(body) = &body {
                request = request.body(body.clone());
            }

            tracing::trace!(%method, url, attempt = retries.attempts(), "sending request");

            let outcome = match request.send().await {
                Ok(response) => {
                    let status = response.status();
                    let response_headers = response.headers().clone();
                    match response.bytes().await {
                        Ok(bytes) => Ok((status, response_headers, bytes)),
                        Err(error) => Err(error),
                    }
                }
                Err(error) => Err(error),
            };

            match outcome {
                Ok((status, response_headers, bytes)) => {
                    let request_id = http::request_id(&response_headers);

                    if status.is_success() {
                        return Ok((bytes.to_vec(), request_id));
                    }

                    let api_error = http::api_error(status.as_u16(), &response_headers, &bytes);

                    match retries.after_status(status.as_u16(), &response_headers) {
                        Some(delay) => {
                            tracing::debug!(
                                status = status.as_u16(),
                                delay_ms = delay.as_millis() as u64,
                                request_id = request_id.as_deref(),
                                "retrying after error response"
                            );
                            tokio::time::sleep(delay).await;
                        }
                        None => return Err(api_error.into()),
                    }
                }
                Err(error) => match retries.after_transport(&error) {
                    Some(delay) => {
                        tracing::debug!(
                            error = %error,
                            delay_ms = delay.as_millis() as u64,
                            "retrying after transport error"
                        );
                        tokio::time::sleep(delay).await;
                    }
                    None => return Err(http::transport_error(error, retries.attempts())),
                },
            }
        }
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.inner.base_url)
            .field("default_model", &self.inner.default_model)
            .field("timeout", &self.inner.timeout)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Options a single call can override.
#[derive(Debug, Default, Clone)]
pub(crate) struct CallOptions {
    pub(crate) timeout: Option<Duration>,
    pub(crate) retry: Option<RetryPolicy>,
    pub(crate) headers: HeaderMap,
    /// Deferred so `header()` can stay infallible and chainable.
    pub(crate) header_error: Option<ConfigError>,
}

impl CallOptions {
    pub(crate) fn add_header(&mut self, name: &str, value: &str) {
        match http::header_pair(name, value) {
            Ok((name, value)) => {
                self.headers.insert(name, value);
            }
            Err(error) => {
                if self.header_error.is_none() {
                    self.header_error = Some(error);
                }
            }
        }
    }
}

/// A pending call to `POST /v1/systemone`.
#[derive(Debug)]
pub struct SystemOneRequest<'a> {
    client: &'a Client,
    questions: &'a QuestionSet,
    state: Result<Value, serde_json::Error>,
    model: Option<Model>,
    options: CallOptions,
}

impl<'a> SystemOneRequest<'a> {
    /// The content to evaluate.
    ///
    /// Anything `Serialize`: a `&str` for plain text, or your own struct for
    /// structured state. Defaults to JSON null when not set.
    ///
    /// A serialization failure is reported by [`send`](Self::send) rather than
    /// here, so calls can chain.
    #[must_use]
    pub fn state<T>(mut self, state: &T) -> Self
    where
        T: Serialize + ?Sized,
    {
        self.state = serde_json::to_value(state);
        self
    }

    /// Override the client's default model for this call.
    #[must_use]
    pub fn model(mut self, model: impl Into<Model>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override the per-attempt timeout for this call.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = Some(timeout);
        self
    }

    /// Override the retry policy for this call.
    #[must_use]
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.options.retry = Some(retry);
        self
    }

    /// Add a header to this call.
    #[must_use]
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.options.add_header(name, value);
        self
    }

    /// Send the request.
    ///
    /// # Errors
    ///
    /// The question set is validated first, so an empty set or a duplicate
    /// name fails without a network round trip.
    pub async fn send(self) -> Result<SystemOne, Error> {
        let SystemOneRequest {
            client,
            questions,
            state,
            model,
            options,
        } = self;

        questions.validate()?;

        let state = state.map_err(|source| Error::Decode {
            source,
            request_id: None,
        })?;

        let model = model.unwrap_or_else(|| client.inner.default_model.clone());
        let body = http::system_one_body(questions, &state, &model)?;
        let url = http::endpoint(&client.inner.base_url, http::SYSTEM_ONE_PATH);

        let (bytes, request_id) = client
            .execute(Method::POST, &url, Some(body), &options)
            .await?;

        let wire: WireResponse =
            serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
                source,
                request_id: request_id.clone(),
            })?;

        SystemOne::from_wire(wire, request_id, questions)
    }
}

/// A pending call to `GET /v1/models`.
#[derive(Debug)]
pub struct ModelsRequest<'a> {
    client: &'a Client,
    options: CallOptions,
}

impl<'a> ModelsRequest<'a> {
    /// Override the per-attempt timeout for this call.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = Some(timeout);
        self
    }

    /// Override the retry policy for this call.
    #[must_use]
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.options.retry = Some(retry);
        self
    }

    /// Add a header to this call.
    #[must_use]
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.options.add_header(name, value);
        self
    }

    /// Send the request.
    pub async fn send(self) -> Result<Vec<ModelCard>, Error> {
        let url = http::endpoint(&self.client.inner.base_url, http::MODELS_PATH);
        let (bytes, request_id) = self
            .client
            .execute(Method::GET, &url, None, &self.options)
            .await?;

        let wire: WireModels = serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
            source,
            request_id,
        })?;

        Ok(wire.models)
    }
}

/// Builder for [`Client`].
///
/// Explicit values win over environment variables, which win over defaults.
#[derive(Debug, Default)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<Model>,
    timeout: Option<Duration>,
    retry: Option<RetryPolicy>,
    default_headers: HeaderMap,
    http: Option<reqwest::Client>,
    error: Option<ConfigError>,
}

impl ClientBuilder {
    /// A builder with nothing set.
    pub fn new() -> Self {
        ClientBuilder::default()
    }

    /// The API key. Falls back to `TYPESAFE_API_KEY`.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// The API root. Falls back to `TYPESAFE_BASE_URL`, then
    /// `https://api.typesafe.ai`.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// The model used when a request does not name one. Falls back to
    /// `TYPESAFE_DEFAULT_MODEL`, then `jev-latest`.
    #[must_use]
    pub fn default_model(mut self, model: impl Into<Model>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// The per-attempt timeout. Defaults to 10 seconds.
    ///
    /// There is no total budget across retries, matching the other SDKs.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The retry policy. Defaults to [`RetryPolicy::default`].
    #[must_use]
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Add a header sent with every request.
    ///
    /// `Authorization` cannot be overridden this way.
    #[must_use]
    pub fn default_header(mut self, name: &str, value: &str) -> Self {
        match http::header_pair(name, value) {
            Ok((name, value)) => {
                self.default_headers.insert(name, value);
            }
            Err(error) => {
                if self.error.is_none() {
                    self.error = Some(error);
                }
            }
        }
        self
    }

    /// Supply a pre-configured `reqwest` client.
    ///
    /// The way to reach transport settings this builder does not expose —
    /// proxies, custom certificates, connection pool tuning.
    #[must_use]
    pub fn http_client(mut self, http: reqwest::Client) -> Self {
        self.http = Some(http);
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<Client, Error> {
        if let Some(error) = self.error {
            return Err(error.into());
        }

        let api_key = self
            .api_key
            .or_else(|| env_var(ENV_API_KEY))
            .ok_or(ConfigError::MissingApiKey)?;

        let base_url = self
            .base_url
            .or_else(|| env_var(ENV_BASE_URL))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let base_url = normalize_base_url(&base_url);
        validate_base_url(&base_url)?;

        let default_model = self
            .default_model
            .or_else(|| env_var(ENV_DEFAULT_MODEL).map(Model::from))
            .unwrap_or_default();

        let http = match self.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .build()
                .map_err(|error| ConfigError::Http(Arc::new(error)))?,
        };

        Ok(Client {
            inner: Arc::new(Inner {
                http,
                base_url,
                api_key,
                default_model,
                default_headers: self.default_headers,
                timeout: self.timeout.unwrap_or(DEFAULT_TIMEOUT),
                retry: self.retry.unwrap_or_default(),
            }),
        })
    }
}

/// Reject a base URL now rather than on every request.
pub(crate) fn validate_base_url(base_url: &str) -> Result<(), ConfigError> {
    let url = Url::parse(base_url).map_err(|error| ConfigError::InvalidBaseUrl {
        url: base_url.to_owned(),
        reason: error.to_string(),
    })?;

    if !matches!(url.scheme(), "http" | "https") {
        return Err(ConfigError::InvalidBaseUrl {
            url: base_url.to_owned(),
            reason: format!("expected an http or https URL, found scheme {:?}", url.scheme()),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_api_key_is_reported() {
        let error = Client::builder().base_url("https://example.test").build();
        // The environment may legitimately carry a key; only assert when it does not.
        if env_var(ENV_API_KEY).is_none() {
            assert!(matches!(
                error,
                Err(Error::Config(ConfigError::MissingApiKey))
            ));
        }
    }

    #[test]
    fn defaults_are_applied() {
        let client = Client::builder().api_key("sk-test").build().unwrap();
        assert_eq!(client.default_model().as_str(), "jev-latest");
        assert_eq!(client.timeout(), DEFAULT_TIMEOUT);
        assert_eq!(client.retry().max_retries, 2);
        // Only meaningful when the environment is not overriding it.
        if env_var(ENV_BASE_URL).is_none() {
            assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        }
    }

    #[test]
    fn trailing_slashes_are_stripped_from_the_base_url() {
        let client = Client::builder()
            .api_key("sk-test")
            .base_url("https://example.test/")
            .build()
            .unwrap();
        assert_eq!(client.base_url(), "https://example.test");
    }

    #[test]
    fn a_nonsense_base_url_is_rejected() {
        let error = Client::builder()
            .api_key("sk-test")
            .base_url("not a url")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Config(ConfigError::InvalidBaseUrl { .. })
        ));
    }

    #[test]
    fn a_non_http_base_url_is_rejected() {
        let error = Client::builder()
            .api_key("sk-test")
            .base_url("ftp://example.test")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Config(ConfigError::InvalidBaseUrl { .. })
        ));
    }

    #[test]
    fn a_bad_default_header_is_rejected() {
        let error = Client::builder()
            .api_key("sk-test")
            .default_header("not a header name", "v")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Config(ConfigError::InvalidHeader { .. })
        ));
    }

    #[test]
    fn debug_output_hides_the_api_key() {
        let client = Client::builder().api_key("sk-super-secret").build().unwrap();
        let rendered = format!("{client:?}");
        assert!(!rendered.contains("sk-super-secret"), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
    }
}
