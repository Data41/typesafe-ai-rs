//! Tests that talk to the real API.
//!
//! Ignored by default so `cargo test` stays hermetic and free. Run them when
//! you want to confirm the crate still matches the live service:
//!
//! ```sh
//! export TYPESAFE_API_KEY=$(security find-generic-password -s typesafe-ai -a TYPESAFE_API_KEY -w)
//! cargo test --test live -- --ignored --test-threads=1
//! ```
//!
//! Each test makes at most one billable call, against a deliberately short
//! state. TypeSafe bills input tokens only, at $0.042 per million, so a full
//! run costs a small fraction of a cent.

use typesafe::{Client, Error, QuestionSet};

fn client() -> Client {
    Client::new().expect("set TYPESAFE_API_KEY to run the live tests")
}

#[tokio::test]
#[ignore = "talks to the live API"]
async fn models_can_be_listed() {
    let models = client().models().send().await.unwrap();

    assert!(!models.is_empty());
    assert!(
        models.iter().any(|model| model.name == "jev-latest"),
        "expected jev-latest in {models:?}"
    );
    assert!(models.iter().all(|model| !model.release_date.is_empty()));
}

#[tokio::test]
#[ignore = "talks to the live API"]
async fn all_three_answer_types_come_back_as_documented() {
    let mut q = QuestionSet::new();
    let is_urgent = q.noul("is_urgent", "Does this convey urgency?");
    let department = q.choice("department", "Which team?", [
        ("billing", "Payments"),
        ("technical", "Bugs"),
        ("sales", "Pricing"),
    ]);
    let frustration = q.score("frustration", "How frustrated?", [
        "Calm",
        "Frustrated",
        "Very angry",
    ]);

    let answers = client()
        .system_one(&q)
        .state("Payouts failing 3 days. Losing sales. Help ASAP.")
        .send()
        .await
        .unwrap();

    // The response reports the versioned id behind the alias.
    assert!(
        answers.model().starts_with("jev-"),
        "unexpected model {:?}",
        answers.model()
    );
    assert!(answers.request_id().is_some(), "missing request id header");
    assert!(answers.usage().input_tokens > 0);

    let urgency = answers[is_urgent].noul;
    assert!((0.0..=1.0).contains(&urgency), "noul out of range: {urgency}");
    assert!(urgency > 0.5, "this ticket is plainly urgent, got {urgency}");

    let choice = &answers[department];
    assert!(choice.probabilities.contains_key(&choice.choice));
    assert_eq!(choice.probabilities.len(), 3);
    let total: f64 = choice.probabilities.values().sum();
    assert!((total - 1.0).abs() < 0.05, "probabilities sum to {total}");
    assert!((0.0..=1.0).contains(&choice.confidence));

    let score = &answers[frustration];
    assert_eq!(score.levels(), 3, "legend should re-index to 3 levels");
    assert_eq!(score.legend_for(0), Some("Calm"));
    assert_eq!(score.legend_for(2), Some("Very angry"));
    // The docs' quickstart omits `probabilities`, but the live API sends it.
    assert_eq!(score.probabilities.len(), 3, "live API sends probabilities");
    assert!((0.0..=2.0).contains(&score.score));
}

#[tokio::test]
#[ignore = "talks to the live API"]
async fn a_bad_api_key_reads_as_an_auth_failure() {
    let client = Client::builder()
        .api_key("apikey_definitely_not_valid")
        .build()
        .unwrap();

    let error = client.models().send().await.unwrap_err();

    match error {
        Error::Api(api) => {
            assert!(api.is_auth(), "expected an auth failure, got {api:?}");
            assert_eq!(api.error_type.as_deref(), Some("authentication_error"));
            // The message must be the server's prose, not a stringified blob.
            assert!(!api.message.contains("error_type"), "{}", api.message);
            assert!(api.message.contains("API key"), "{}", api.message);
        }
        other => panic!("expected an API error, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "talks to the live API"]
async fn an_empty_api_key_still_reads_as_an_auth_failure() {
    // The API answers a missing credential with 403 rather than 401; is_auth
    // has to see through that.
    let client = Client::builder().api_key(" x").build().unwrap();
    let error = client.models().send().await.unwrap_err();

    match error {
        Error::Api(api) => assert!(
            api.is_auth(),
            "status {} / type {:?} should read as auth",
            api.status,
            api.error_type
        ),
        other => panic!("expected an API error, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "talks to the live API"]
async fn an_unknown_model_is_a_bad_request() {
    let mut q = QuestionSet::new();
    q.noul("q", "?");

    let error = client()
        .system_one(&q)
        .model("definitely-not-a-model")
        .state("x")
        .send()
        .await
        .unwrap_err();

    match error {
        Error::Api(api) => {
            assert!(api.is_bad_request(), "got status {}", api.status);
            assert!(!api.is_auth());
            assert!(
                api.message.to_lowercase().contains("model"),
                "{}",
                api.message
            );
        }
        other => panic!("expected an API error, got {other:?}"),
    }
}
