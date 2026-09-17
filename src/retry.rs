//! Retry policy and backoff.
//!
//! TypeSafe's docs are explicit that rate limits "can change without notice"
//! while they scale, and that a `429` or `529` should be retried with
//! exponential backoff. The defaults here mirror the official SDKs.

use std::ops::RangeInclusive;
use std::time::Duration;

use reqwest::header::HeaderMap;

/// When and how to retry a failed request.
///
/// The defaults match the JavaScript and Python SDKs: two retries, backoff
/// from 500ms doubling to a 5s ceiling with 25% jitter, retrying `408`, `429`
/// and `5xx` plus connection errors and timeouts, honouring `Retry-After` up
/// to a minute.
///
/// # Adjusting it
///
/// The struct is `#[non_exhaustive]`, so new knobs can be added without a
/// breaking release. That rules out `..Default::default()` construction, but
/// the fields are public and the chainable setters cover the common cases:
///
/// ```
/// use std::time::Duration;
/// use typesafe::RetryPolicy;
///
/// // Give up faster, and never wait more than a second between tries.
/// let policy = RetryPolicy::default()
///     .with_max_retries(1)
///     .with_backoff_max(Duration::from_secs(1));
///
/// // Or mutate in place, which works for every field.
/// let mut policy = RetryPolicy::default();
/// policy.backoff_jitter = 0.5;
/// policy.retry_statuses.push(418..=418);
/// # let _ = policy;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Retries *after* the first attempt. `0` disables retrying.
    pub max_retries: u32,
    /// The first backoff delay, doubled on each retry up to `backoff_max`.
    pub backoff_initial: Duration,
    /// Ceiling on the backoff delay.
    pub backoff_max: Duration,
    /// Fraction of each delay randomly subtracted, from 0 to 1.
    ///
    /// Jitter is subtractive rather than additive so a delay never exceeds
    /// `backoff_max`.
    pub backoff_jitter: f64,
    /// Whether to honour `Retry-After` and `retry-after-ms`.
    pub respect_retry_after: bool,
    /// Ceiling on a server-requested delay. Longer ones fall back to backoff.
    pub max_retry_after: Duration,
    /// Whether to retry connection failures.
    pub retry_connection_errors: bool,
    /// Whether to retry per-attempt timeouts.
    pub retry_timeouts: bool,
    /// Status codes to retry.
    pub retry_statuses: Vec<RangeInclusive<u16>>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_retries: 2,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(5),
            backoff_jitter: 0.25,
            respect_retry_after: true,
            max_retry_after: Duration::from_secs(60),
            retry_connection_errors: true,
            retry_timeouts: true,
            retry_statuses: vec![408..=408, 429..=429, 500..=599],
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    pub fn none() -> Self {
        RetryPolicy {
            max_retries: 0,
            ..RetryPolicy::default()
        }
    }

    /// Set the number of retries after the first attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the first backoff delay.
    #[must_use]
    pub fn with_backoff_initial(mut self, backoff_initial: Duration) -> Self {
        self.backoff_initial = backoff_initial;
        self
    }

    /// Set the ceiling on the backoff delay.
    #[must_use]
    pub fn with_backoff_max(mut self, backoff_max: Duration) -> Self {
        self.backoff_max = backoff_max;
        self
    }

    /// Set the jitter fraction, from 0 to 1.
    #[must_use]
    pub fn with_backoff_jitter(mut self, backoff_jitter: f64) -> Self {
        self.backoff_jitter = backoff_jitter;
        self
    }

    /// Set whether `Retry-After` is honoured.
    #[must_use]
    pub fn with_respect_retry_after(mut self, respect: bool) -> Self {
        self.respect_retry_after = respect;
        self
    }

    /// Set the ceiling on a server-requested delay.
    #[must_use]
    pub fn with_max_retry_after(mut self, max_retry_after: Duration) -> Self {
        self.max_retry_after = max_retry_after;
        self
    }

    /// Replace the set of retryable status codes.
    #[must_use]
    pub fn with_retry_statuses(
        mut self,
        statuses: impl IntoIterator<Item = RangeInclusive<u16>>,
    ) -> Self {
        self.retry_statuses = statuses.into_iter().collect();
        self
    }

    /// Whether a response with this status should be retried.
    pub fn retries_status(&self, status: u16) -> bool {
        self.retry_statuses
            .iter()
            .any(|range| range.contains(&status))
    }

    /// How long to wait before retry number `retry` (0 for the first retry).
    ///
    /// `server_delay` is what the response asked for, if anything; it wins
    /// over backoff when it is honoured and within `max_retry_after`.
    pub(crate) fn delay(&self, retry: u32, server_delay: Option<Duration>) -> Duration {
        if self.respect_retry_after
            && let Some(delay) = server_delay
            && delay <= self.max_retry_after
        {
            return delay;
        }
        self.backoff(retry)
    }

    fn backoff(&self, retry: u32) -> Duration {
        let initial = self.backoff_initial.as_millis() as u64;
        let ceiling = self.backoff_max.as_millis() as u64;

        // Saturating, and the shift is capped, so a large retry count cannot
        // overflow into a short delay.
        let doubled = initial.saturating_mul(1u64 << retry.min(32));
        let capped = doubled.min(ceiling);

        let jitter = self.backoff_jitter.clamp(0.0, 1.0);
        let reduction = capped as f64 * jitter * fastrand::f64();
        Duration::from_millis((capped as f64 - reduction).round() as u64)
    }
}

/// The delay a response asked for, from `retry-after-ms` or `Retry-After`.
///
/// `retry-after-ms` wins when both are present: it is the more precise of the
/// two, and is what the SDKs check first.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok())
        && let Ok(millis) = value.trim().parse::<f64>()
        && millis.is_finite()
        && millis >= 0.0
    {
        return Some(Duration::from_millis(millis.round() as u64));
    }

    let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let value = value.trim();

    // `Retry-After` is either delay-seconds or an HTTP-date.
    if let Ok(seconds) = value.parse::<f64>() {
        if seconds.is_finite() && seconds >= 0.0 {
            return Some(Duration::from_secs_f64(seconds));
        }
        return None;
    }

    let deadline = httpdate::parse_http_date(value).ok()?;
    deadline
        .duration_since(std::time::SystemTime::now())
        .ok()
        .or(Some(Duration::ZERO))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        headers
    }

    #[test]
    fn default_statuses_match_the_documented_set() {
        let policy = RetryPolicy::default();
        for status in [408, 429, 500, 503, 529, 599] {
            assert!(policy.retries_status(status), "expected {status} to retry");
        }
        for status in [200, 400, 401, 404, 422, 600] {
            assert!(!policy.retries_status(status), "expected {status} to stop");
        }
    }

    #[test]
    fn backoff_doubles_and_stays_within_bounds() {
        let policy = RetryPolicy::default();
        for retry in 0..8 {
            let delay = policy.delay(retry, None);
            let uncapped = 500u64.saturating_mul(1 << retry.min(32));
            let expected = uncapped.min(5_000);
            // Jitter only ever subtracts, and never more than 25%.
            assert!(
                delay.as_millis() as u64 <= expected,
                "retry {retry}: {delay:?} exceeded {expected}ms"
            );
            assert!(
                delay.as_millis() as f64 >= expected as f64 * 0.74,
                "retry {retry}: {delay:?} below the jitter floor"
            );
        }
    }

    #[test]
    fn backoff_does_not_overflow_at_absurd_retry_counts() {
        let policy = RetryPolicy::default();
        assert!(policy.delay(u32::MAX, None) <= Duration::from_secs(5));
    }

    #[test]
    fn zero_jitter_is_exact() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        assert_eq!(policy.delay(0, None), Duration::from_millis(500));
        assert_eq!(policy.delay(1, None), Duration::from_millis(1000));
        assert_eq!(policy.delay(4, None), Duration::from_millis(5000));
    }

    #[test]
    fn retry_after_seconds_are_honoured() {
        let policy = RetryPolicy::default();
        let delay = retry_after(&headers(&[("retry-after", "3")]));
        assert_eq!(delay, Some(Duration::from_secs(3)));
        assert_eq!(policy.delay(0, delay), Duration::from_secs(3));
    }

    #[test]
    fn retry_after_ms_wins_over_retry_after() {
        let delay = retry_after(&headers(&[("retry-after", "3"), ("retry-after-ms", "250")]));
        assert_eq!(delay, Some(Duration::from_millis(250)));
    }

    #[test]
    fn an_http_date_in_the_past_means_no_delay() {
        let delay = retry_after(&headers(&[(
            "retry-after",
            "Wed, 21 Oct 2015 07:28:00 GMT",
        )]));
        assert_eq!(delay, Some(Duration::ZERO));
    }

    #[test]
    fn an_http_date_in_the_future_is_a_positive_delay() {
        let future = std::time::SystemTime::now() + Duration::from_secs(120);
        let delay = retry_after(&headers(&[(
            "retry-after",
            &httpdate::fmt_http_date(future),
        )]))
        .unwrap();
        assert!(delay > Duration::from_secs(60), "{delay:?}");
        assert!(delay <= Duration::from_secs(120), "{delay:?}");
    }

    #[test]
    fn an_overlong_server_delay_falls_back_to_backoff() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        let delay = policy.delay(0, Some(Duration::from_secs(600)));
        assert_eq!(delay, Duration::from_millis(500));
    }

    #[test]
    fn a_server_delay_is_ignored_when_not_respected() {
        let policy = RetryPolicy {
            respect_retry_after: false,
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        assert_eq!(
            policy.delay(0, Some(Duration::from_secs(2))),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn setters_chain() {
        let policy = RetryPolicy::default()
            .with_max_retries(5)
            .with_backoff_max(Duration::from_secs(1))
            .with_backoff_jitter(0.0)
            .with_retry_statuses([429..=429]);
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.delay(9, None), Duration::from_secs(1));
        assert!(policy.retries_status(429));
        assert!(!policy.retries_status(500));
        assert_eq!(RetryPolicy::none().max_retries, 0);
    }

    #[test]
    fn garbage_retry_after_values_are_ignored() {
        assert_eq!(retry_after(&headers(&[("retry-after", "soon")])), None);
        assert_eq!(retry_after(&headers(&[("retry-after", "-5")])), None);
        assert_eq!(retry_after(&HeaderMap::new()), None);
    }
}
