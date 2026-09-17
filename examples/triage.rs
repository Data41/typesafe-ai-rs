//! Classify one support ticket with three questions in a single call.
//!
//! This is the quickstart from <https://docs.typesafe.ai/introduction/quickstart>.
//!
//! ```sh
//! export TYPESAFE_API_KEY=sk-...
//! cargo run --example triage
//! ```

use typesafe::{Client, QuestionSet};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new()?;

    // One set, three questions, one round trip. Every question is evaluated in
    // parallel, so asking a question you might not need is close to free.
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

    let is_urgent = q.noul(
        "is_urgent",
        "The message conveys urgency or time-sensitivity",
    );

    let ticket = "Hi, I've been trying to connect my Stripe account for 3 days \
                  and it keeps failing. I'm losing sales. Please help ASAP.";

    let answers = client.system_one(&q).state(ticket).send().await?;

    // Each key reads back at its own type: f64, ChoiceAnswer, ScoreAnswer.
    println!("model:     {}", answers.model());
    println!("urgency:   {:.1}%", answers[is_urgent].noul * 100.0);
    println!(
        "team:      {} ({:.0}% confident)",
        answers[department].choice,
        answers[department].confidence * 100.0
    );
    println!(
        "frustration: {:.2} — closest level: {:?}",
        answers[frustration].score,
        answers[frustration].legend_for(answers[frustration].nearest_level()),
    );

    // Confidence is a second axis: the answer says what, confidence says
    // whether to act on it without a human.
    match answers[department].choice_above(0.8) {
        Some(team) => println!("\nrouting automatically to {team}"),
        None => println!("\ntoo close to call — queueing for manual triage"),
    }

    println!(
        "\nusage: {} input tokens, {} output tokens",
        answers.usage().input_tokens,
        answers.usage().output_tokens
    );

    Ok(())
}
