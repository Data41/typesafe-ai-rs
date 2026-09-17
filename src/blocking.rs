//! The synchronous client, behind the `blocking` feature.
//!
//! A mirror of [`crate::Client`] with the `await`s removed. It is also the
//! supported path for non-tokio runtimes: `reqwest::blocking` runs its own
//! tokio runtime on a background thread, so `async-std` and `smol` programs
//! can use this without hosting a tokio reactor themselves.
//!
//! Do not call it from inside an async task — it blocks the calling thread.
//! From tokio, reach for [`crate::Client`], or wrap the call in
//! `tokio::task::spawn_blocking`.
//!
//! ```no_run
//! use typesafe::{QuestionSet, blocking::Client};
//!
//! # fn main() -> Result<(), typesafe::Error> {
//! let client = Client::new()?;
//!
//! let mut q = QuestionSet::new();
//! let urgent = q.noul("is_urgent", "Does this convey urgency?");
//!
//! let answers = client.system_one(&q).state("Please help, ASAP!").send()?;
//! println!("urgency: {:.2}", answers[urgent].noul);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

use reqwest::Method;
use reqwest::header::HeaderMap;
use serde::Serialize;
use serde_json::Value;

use crate::client::{CallOptions, validate_base_url};
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

/// A synchronous client for the TypeSafe AI API.
///
/// See the [module docs](self) for when to prefer this over
/// [`crate::Client`].
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
    default_model: Model,
    default_headers: HeaderMap,
    timeout: Duration,
    retry: RetryPolicy,
}

impl Client {
    /// Build a client from the environment. See [`crate::Client::new`].
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

    /// The blocking twin of [`crate::Client::execute`]. The retry decisions
    /// come from the same shared helpers; only the sleep differs.
    fn execute(
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

            let outcome = match request.send() {
                Ok(response) => {
                    let status = response.status();
                    let response_headers = response.headers().clone();
                    match response.bytes() {
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
                            std::thread::sleep(delay);
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
                        std::thread::sleep(delay);
                    }
                    None => return Err(http::transport_error(error, retries.attempts())),
                },
            }
        }
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("blocking::Client")
            .field("base_url", &self.inner.base_url)
            .field("default_model", &self.inner.default_model)
            .field("timeout", &self.inner.timeout)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
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
    /// The content to evaluate. See [`crate::SystemOneRequest::state`].
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

    /// Send the request, blocking until it resolves.
    pub fn send(self) -> Result<SystemOne, Error> {
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

        let (bytes, request_id) = client.execute(Method::POST, &url, Some(body), &options)?;

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

    /// Send the request, blocking until it resolves.
    pub fn send(self) -> Result<Vec<ModelCard>, Error> {
        let url = http::endpoint(&self.client.inner.base_url, http::MODELS_PATH);
        let (bytes, request_id) = self.client.execute(Method::GET, &url, None, &self.options)?;

        let wire: WireModels = serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
            source,
            request_id,
        })?;

        Ok(wire.models)
    }
}

/// Builder for the blocking [`Client`].
#[derive(Debug, Default)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<Model>,
    timeout: Option<Duration>,
    retry: Option<RetryPolicy>,
    default_headers: HeaderMap,
    http: Option<reqwest::blocking::Client>,
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

    /// The API root. Falls back to `TYPESAFE_BASE_URL`.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// The default model. Falls back to `TYPESAFE_DEFAULT_MODEL`.
    #[must_use]
    pub fn default_model(mut self, model: impl Into<Model>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// The per-attempt timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The retry policy.
    #[must_use]
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Add a header sent with every request.
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

    /// Supply a pre-configured blocking `reqwest` client.
    #[must_use]
    pub fn http_client(mut self, http: reqwest::blocking::Client) -> Self {
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
            None => reqwest::blocking::Client::builder()
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
