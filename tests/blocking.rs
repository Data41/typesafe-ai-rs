//! End-to-end tests for the synchronous client.

#![cfg(feature = "blocking")]

mod support;

use support::{Canned, MockServer, QUICKSTART_RESPONSE};
use typesafe::blocking::Client;
use typesafe::{Error, ProtocolError, QuestionSet, RetryPolicy};

fn client(server: &MockServer) -> Client {
    Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .retry(RetryPolicy::none())
        .build()
        .unwrap()
}

#[test]
fn the_documented_exchange_round_trips() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);

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
    let is_urgent = q.noul("is_urgent", "The message conveys urgency");

    let answers = client(&server)
        .system_one(&q)
        .state("Help! My payouts have been failing for 3 days.")
        .send()
        .unwrap();

    assert_eq!(answers[is_urgent].noul, 0.999);
    assert_eq!(answers[department].choice, "billing");
    assert_eq!(answers[frustration].score, 1.035);
    assert_eq!(answers.model(), "jev-1.13.0");
    assert_eq!(answers.usage().input_tokens, 312);

    let request = &server.requests()[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/systemone");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
}

#[test]
fn models_are_listed() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"models":[{"name":"jev-latest","description":"Stable","release_date":"2026-08-01"}]}"#,
    )]);

    let models = client(&server).models().send().unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "jev-latest");
}

#[test]
fn errors_map_the_same_way_as_the_async_client() {
    let server = MockServer::start(vec![Canned::json(401, r#"{"message":"invalid api key"}"#)]);
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let error = client(&server).system_one(&q).send().unwrap_err();

    match error {
        Error::Api(api) => {
            assert!(api.is_auth());
            assert_eq!(api.message, "invalid api key");
        }
        other => panic!("expected an API error, got {other:?}"),
    }
}

#[test]
fn a_rate_limited_request_is_retried_and_then_succeeds() {
    let server = MockServer::start(vec![
        Canned::json(429, r#"{"message":"slow down"}"#).header("retry-after-ms", "20"),
        Canned::ok(r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.7}},"usage":{}}"#),
    ]);
    let mut q = QuestionSet::new();
    let key = q.noul("q", "?");

    let client = Client::builder()
        .api_key("sk-test")
        .base_url(&server.url)
        .build()
        .unwrap();

    let answers = client.system_one(&q).send().unwrap();

    assert_eq!(answers[key].noul, 0.7);
    assert_eq!(server.hits(), 2, "expected exactly one retry");
}

#[test]
fn response_validation_matches_the_async_client() {
    let server = MockServer::start(vec![Canned::ok(
        r#"{"model":"m","answers":{"a":{"type":"noul","noul":0.5}},"usage":{}}"#,
    )]);
    let mut q = QuestionSet::new();
    q.noul("a", "?");
    q.noul("b", "?");

    let error = client(&server).system_one(&q).send().unwrap_err();

    assert!(
        matches!(
            &error,
            Error::Protocol(ProtocolError::MissingAnswer { name }) if name == "b"
        ),
        "{error:?}"
    );
}

#[test]
fn an_empty_question_set_never_reaches_the_network() {
    let server = MockServer::start(vec![Canned::ok(QUICKSTART_RESPONSE)]);
    let q = QuestionSet::new();

    client(&server).system_one(&q).send().unwrap_err();
    assert_eq!(server.hits(), 0);
}
