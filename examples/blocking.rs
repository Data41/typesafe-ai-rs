//! The synchronous client — also the path for non-tokio runtimes.
//!
//! `reqwest::blocking` runs its own tokio reactor on a background thread, so
//! this works from `async-std`, `smol`, or a plain `fn main`.
//!
//! ```sh
//! export TYPESAFE_API_KEY=sk-...
//! cargo run --example blocking --features blocking
//! ```

use typesafe::QuestionSet;
use typesafe::blocking::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new()?;

    println!("models available to this account:");
    for model in client.models().send()? {
        println!("  {:<14} {}  {}", model.name, model.release_date, model.description);
    }

    let mut q = QuestionSet::new();
    let refundable = q.noul_with(
        "refundable",
        "Is this request eligible for a refund?",
        "The policy clearly covers this situation",
        "The policy does not cover it, or there is not enough information",
    );

    // Structured state: the model reads labelled fields rather than a string
    // template you glued together.
    let state = serde_json::json!({
        "ticket_message": "My flight was cancelled. Can I get a refund?",
        "refund_policy": "Cancelled flights are eligible for a full refund.",
    });

    let answers = client.system_one(&q).state(&state).send()?;

    println!("\nrefundable: {:.1}%", answers[refundable].noul * 100.0);
    Ok(())
}
