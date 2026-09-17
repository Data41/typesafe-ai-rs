//! End-to-end tests against a canned HTTP server.

mod support;

use std::time::{Duration, Instant};

use serde_json::json;
use support::{Canned, MockServer, QUICKSTART_RESPONSE};
use typesafe::{Client, Error, KeyError, ProtocolError, QuestionSet, RetryPolicy};

fn client(server: &MockServer) -> Client {
    Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .retry(RetryPolicy::none())
        .build()
        .unwrap()
}

/// The three-question set from the quickstart.
fn quickstart_questions() -> (
    QuestionSet,
    typesafe::ChoiceKey,
    typesafe::ScoreKey,
    typesafe::NoulKey,
) {
    let mut q = QuestionSet::new();
    let department = q.choice("department", "Which team should handle this", [
        ("billing", "Payment or subscription issues"),
        ("technical", "Bugs or integration problems"),
        ("sales", "Pricing or account questions"),
    ]);
    let frustration = q.score("frustration", "How frustrated the customer appears", [
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, strong language",
    ]);
    let is_urgent = q.noul("is_urgent", "The message conveys urgency or time-sensitivity");
    (q, department, frustration, is_urgent)
}

#[tokio::test]
async fn the_documented_exchange_round_trips() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);
    let (q, department, frustration, is_urgent) = quickstart_questions();

    let answers = client(&server)
        .system_one(&q)
        .state("Hi, I've been trying to connect my Stripe account for 3 days.")
        .send()
        .await
        .unwrap();

    // Typed reads, no match and no unwrap.
    assert_eq!(answers[is_urgent].noul, 0.999);
    assert_eq!(answers[department].choice, "billing");
    assert_eq!(answers[department].confidence, 0.596);
    assert_eq!(answers[frustration].score, 1.035);

    assert_eq!(answers.model(), "jev-1.13.0");
    assert_eq!(answers.usage().input_tokens, 312);
    assert_eq!(answers.usage().output_tokens, 48);
    assert_eq!(answers.len(), 3);

    // Score levels are re-indexed from the "0"/"1"/"2" object.
    assert_eq!(answers[frustration].legend_for(1), Some("Frustrated but civil"));
    assert_eq!(answers[frustration].nearest_level(), 1);

    // Choice helpers.
    assert_eq!(answers[department].probability("technical"), Some(0.159));
    assert_eq!(answers[department].ranked()[0].0, "billing");

    // Untyped escape hatches still work.
    assert_eq!(answers.by_name("is_urgent").unwrap().tag(), "noul");
    assert_eq!(answers.iter().count(), 3);
}

#[tokio::test]
async fn the_request_carries_auth_and_the_documented_body() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);
    let (q, ..) = quickstart_questions();

    client(&server)
        .system_one(&q)
        .state("Help!")
        .send()
        .await
        .unwrap();

    let request = &server.requests()[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/systemone");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert!(
        request.header("user-agent").unwrap().starts_with("typesafe-ai-rs/"),
        "{:?}",
        request.header("user-agent")
    );

    let body = request.json();
    assert_eq!(body["state"], json!("Help!"));
    assert_eq!(body["model"], json!("jev-latest"));
    assert_eq!(body["questions"]["is_urgent"]["type"], json!("noul"));
    assert_eq!(
        body["questions"]["frustration"]["criteria"],
        json!([
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language"
        ])
    );
}

#[tokio::test]
async fn structured_state_is_sent_as_json() {
    #[derive(serde::Serialize)]
    struct Ticket<'a> {
        ticket_message: &'a str,
        refund_policy: &'a str,
    }

    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    let key = q.noul("q", "Eligible?");

    let answers = client(&server)
        .system_one(&q)
        .state(&Ticket {
            ticket_message: "My flight was cancelled.",
            refund_policy: "Cancelled flights get a full refund.",
        })
        .send()
        .await
        .unwrap();

    assert_eq!(answers[key].noul, 0.5);
    assert_eq!(
        server.requests()[0].json()["state"],
        json!({
            "ticket_message": "My flight was cancelled.",
            "refund_policy": "Cancelled flights get a full refund."
        })
    );
}

#[tokio::test]
async fn the_model_can_be_pinned_per_call() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.1}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    client(&server)
        .system_one(&q)
        .model("jev-1.13.0")
        .send()
        .await
        .unwrap();

    assert_eq!(server.requests()[0].json()["model"], json!("jev-1.13.0"));
}

#[tokio::test]
async fn models_are_listed() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"models":[
            {"name":"jev-latest","description":"Most recent stable release","release_date":"2026-08-01"},
            {"name":"jev-preview","description":"Most recent release","release_date":"2026-08-01"}
        ]}"#,
    )]);

    let models = client(&server).models().send().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name, "jev-latest");
    assert_eq!(models[0].release_date, "2026-08-01");
    assert_eq!(server.requests()[0].method, "GET");
    assert_eq!(server.requests()[0].path, "/v1/models");
}

// --- Error mapping -------------------------------------------------------

async fn api_error_for(status: u16, body: &str) -> typesafe::ApiError {
    let server = MockServer::start(vec![Canned::json(status, body)]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let error = client(&server).system_one(&q).send().await.unwrap_err();
    match error {
        Error::Api(api) => *api,
        other => panic!("expected an API error, got {other:?}"),
    }
}

#[tokio::test]
async fn documented_statuses_map_to_predicates() {
    let unauthorized = api_error_for(401, r#"{"message":"invalid api key"}"#).await;
    assert!(unauthorized.is_auth());
    assert_eq!(unauthorized.message, "invalid api key");

    let unprocessable = api_error_for(422, r#"{"detail":"questions must not be empty"}"#).await;
    assert!(unprocessable.is_unprocessable());
    assert_eq!(unprocessable.message, "questions must not be empty");

    let rate_limited = api_error_for(429, r#"{"message":"slow down"}"#).await;
    assert!(rate_limited.is_rate_limit());

    let overloaded = api_error_for(529, r#"{"message":"overloaded"}"#).await;
    assert!(overloaded.is_overloaded());

    let server_error = api_error_for(500, r#"{"message":"boom"}"#).await;
    assert!(server_error.is_server_error());
}

#[tokio::test]
async fn the_request_id_header_is_surfaced() {
    let server = MockServer::start(vec![
        Canned::json(401, r#"{"message":"nope"}"#).header("x-typesafe-request-id", "req_abc123"),
    ]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let error = client(&server).system_one(&q).send().await.unwrap_err();
    assert_eq!(error.request_id(), Some("req_abc123"));
}

#[tokio::test]
async fn the_request_id_is_surfaced_on_success_too() {
    let server = MockServer::start(vec![
        Canned::ok(r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.2}},"usage":{}}"#)
            .header("x-typesafe-request-id", "req_ok"),
    ]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let answers = client(&server).system_one(&q).send().await.unwrap();
    assert_eq!(answers.request_id(), Some("req_ok"));
}

// --- Retries -------------------------------------------------------------

#[tokio::test]
async fn a_rate_limited_request_is_retried_and_then_succeeds() {
    let server = MockServer::start(vec![
        Canned::json(429, r#"{"message":"slow down"}"#).header("retry-after-ms", "20"),
        Canned::ok(r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.7}},"usage":{}}"#),
    ]);
    let mut q = QuestionSet::new();
    let key = q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .build()
        .unwrap();

    let answers = client.system_one(&q).send().await.unwrap();

    assert_eq!(answers[key].noul, 0.7);
    assert_eq!(server.hits(), 2, "expected exactly one retry");
}

#[tokio::test]
async fn retries_are_bounded_by_the_policy() {
    let server = MockServer::start(vec![
        Canned::json(529, r#"{"message":"overloaded"}"#).header("retry-after-ms", "5"),
    ]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .retry(RetryPolicy::default().with_max_retries(2))
        .build()
        .unwrap();

    let error = client.system_one(&q).send().await.unwrap_err();

    assert!(error.is_transient());
    // First attempt plus two retries.
    assert_eq!(server.hits(), 3);
}

#[tokio::test]
async fn a_retry_after_header_is_actually_waited_out() {
    let server = MockServer::start(vec![
        Canned::json(429, "{}").header("retry-after-ms", "400"),
        Canned::ok(r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.1}},"usage":{}}"#),
    ]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        // Backoff alone would be ~500ms with jitter; the header asks for 400ms,
        // and respecting it is what we are checking.
        .retry(RetryPolicy::default().with_max_retries(1))
        .build()
        .unwrap();

    let started = Instant::now();
    client.system_one(&q).send().await.unwrap();
    let elapsed = started.elapsed();

    assert!(
        elapsed >= Duration::from_millis(350),
        "returned too fast to have waited: {elapsed:?}"
    );
}

#[tokio::test]
async fn non_retryable_statuses_are_not_retried() {
    let server = MockServer::start(vec![Canned::json(422, r#"{"message":"bad"}"#)]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .build()
        .unwrap();

    client.system_one(&q).send().await.unwrap_err();
    assert_eq!(server.hits(), 1);
}

// --- Protocol validation -------------------------------------------------

#[tokio::test]
async fn an_unanswered_question_is_an_error_not_a_panic() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"a":{"type":"noul","noul":0.5}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.noul("a", "?");
    q.noul("b", "?");

    let error = client(&server).system_one(&q).send().await.unwrap_err();

    assert!(
        matches!(
            &error,
            Error::Protocol(ProtocolError::MissingAnswer { name }) if name == "b"
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn an_answer_of_the_wrong_type_is_rejected() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"a":{"type":"noul","noul":0.5}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.score("a", "?", ["low", "high"]);

    let error = client(&server).system_one(&q).send().await.unwrap_err();

    assert!(
        matches!(
            &error,
            Error::Protocol(ProtocolError::WrongAnswerType { name, expected, .. })
                if name == "a" && *expected == "score"
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn extra_answers_we_did_not_ask_for_are_ignored() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{
            "a":{"type":"noul","noul":0.5},
            "speculative":{"type":"noul","noul":0.9}
        },"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    let a = q.noul("a", "?");

    let answers = client(&server).system_one(&q).send().await.unwrap();

    assert_eq!(answers[a].noul, 0.5);
    assert_eq!(answers.len(), 1);
    assert!(answers.by_name("speculative").is_none());
}

#[tokio::test]
async fn score_levels_beyond_nine_stay_in_numeric_order() {
    // "10" sorts between "1" and "2" as a string. This is the regression guard.
    let legend: String = (0..12)
        .map(|level| format!(r#""{level}":"level {level}""#))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"{{"model":"m","answers":{{"s":{{"type":"score","score":10.0,"legend":{{{legend}}},"confidence":0.9}}}},"usage":{{}}}}"#
    );

    let server = MockServer::start(vec![Canned::ok(&body)]);
    let mut q = QuestionSet::new();
    let s = q.score_iter("s", "?", (0..12).map(|i| format!("level {i}")));

    let answers = client(&server).system_one(&q).send().await.unwrap();

    assert_eq!(answers[s].levels(), 12);
    assert_eq!(answers[s].legend_for(10), Some("level 10"));
    assert_eq!(answers[s].legend_for(2), Some("level 2"));
    assert_eq!(answers[s].nearest_level(), 10);
}

#[tokio::test]
async fn malformed_score_levels_are_rejected() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"s":{"type":"score","score":1.0,
            "legend":{"0":"a","2":"c"},"confidence":0.9}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.score("s", "?", ["a", "b"]);

    let error = client(&server).system_one(&q).send().await.unwrap_err();
    assert!(
        matches!(
            error,
            Error::Protocol(ProtocolError::MalformedScoreKeys { .. })
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_body_that_is_not_json_is_a_decode_error() {
    let server = MockServer::start(vec![Canned::ok("not json at all")]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let error = client(&server).system_one(&q).send().await.unwrap_err();
    assert!(matches!(error, Error::Decode { .. }), "{error:?}");
}

// --- Keys ----------------------------------------------------------------

#[tokio::test]
async fn a_key_from_another_question_set_is_refused() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.4}},"usage":{}}"#,
    )]);

    let mut mine = QuestionSet::new();
    mine.noul("q", "?");

    let mut theirs = QuestionSet::new();
    let foreign = theirs.noul("q", "?");

    let answers = client(&server).system_one(&mine).send().await.unwrap();

    assert!(matches!(
        answers.get(foreign),
        Err(KeyError::ForeignQuestionSet { .. })
    ));
}

#[tokio::test]
#[should_panic(expected = "different QuestionSet")]
async fn indexing_with_a_foreign_key_panics_loudly() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.4}},"usage":{}}"#,
    )]);

    let mut mine = QuestionSet::new();
    mine.noul("q", "?");
    let mut theirs = QuestionSet::new();
    let foreign = theirs.noul("q", "?");

    let answers = client(&server).system_one(&mine).send().await.unwrap();
    let _ = answers[foreign];
}

#[tokio::test]
async fn one_question_set_can_be_reused_across_calls() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.3}},"usage":{}}"#,
    )]);

    let mut q = QuestionSet::new();
    let key = q.noul("q", "?");
    let client = client(&server);

    for state in ["first", "second", "third"] {
        let answers = client.system_one(&q).state(state).send().await.unwrap();
        assert_eq!(answers[key].noul, 0.3);
    }

    assert_eq!(server.hits(), 3);
}

// --- Client-side validation ---------------------------------------------

#[tokio::test]
async fn an_empty_question_set_never_reaches_the_network() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);
    let q = QuestionSet::new();

    let error = client(&server).system_one(&q).send().await.unwrap_err();

    assert!(matches!(
        error,
        Error::QuestionSet(typesafe::QuestionSetError::Empty)
    ));
    assert_eq!(server.hits(), 0, "nothing should have been sent");
}

#[tokio::test]
async fn a_duplicate_question_name_never_reaches_the_network() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);
    let mut q = QuestionSet::new();
    q.noul("dupe", "?");
    q.score("dupe", "?", ["a", "b"]);

    let error = client(&server).system_one(&q).send().await.unwrap_err();

    assert!(matches!(
        error,
        Error::QuestionSet(typesafe::QuestionSetError::DuplicateName { .. })
    ));
    assert_eq!(server.hits(), 0);
}

// --- Per-call options ----------------------------------------------------

#[tokio::test]
async fn per_call_headers_are_sent() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.1}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .default_header("x-tenant", "acme")
        .build()
        .unwrap();

    client
        .system_one(&q)
        .header("x-trace", "abc")
        .send()
        .await
        .unwrap();

    let request = &server.requests()[0];
    assert_eq!(request.header("x-tenant"), Some("acme"));
    assert_eq!(request.header("x-trace"), Some("abc"));
    // A default header must never be able to displace the API key.
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
}

#[tokio::test]
async fn a_default_header_cannot_override_authorization() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.1}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-real")
        .base_url(&server.url)
        .default_header("authorization", "Bearer sk-attacker")
        .build()
        .unwrap();

    client.system_one(&q).send().await.unwrap();
    assert_eq!(
        server.requests()[0].header("authorization"),
        Some("Bearer sk-real")
    );
}
