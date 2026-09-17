//! Build one question set and reuse it across many inputs.
//!
//! Keys are `Copy` and stay valid for the life of the set, so the set and its
//! keys live together in a struct and every ticket reuses them.
//!
//! ```sh
//! export TYPESAFE_API_KEY=sk-...
//! cargo run --example fan_out
//! ```

use std::sync::Arc;

use typesafe::{Client, NoulKey, QuestionSet, ScoreKey};

/// The question set, its keys, and the client — assembled once.
///
/// Wrapping the set in an `Arc` also freezes it: `&mut self` is unreachable
/// through a shared pointer, so no question can be appended behind the keys'
/// backs.
struct Triage {
    client: Client,
    questions: Arc<QuestionSet>,
    is_urgent: NoulKey,
    is_churn_risk: NoulKey,
    severity: ScoreKey,
}

impl Triage {
    fn new(client: Client) -> Self {
        let mut q = QuestionSet::new();

        let is_urgent = q.noul_with(
            "is_urgent",
            "Does this message convey urgency?",
            "Explicitly time-sensitive, or blocking the customer's work",
            "No urgency expressed",
        );

        // A speculative question: cheap enough to always ask, even though most
        // tickets will not need it.
        let is_churn_risk = q.noul(
            "is_churn_risk",
            "Does this customer sound like they might cancel?",
        );

        let severity = q.score("severity", "How severe is the reported problem?", [
            "Cosmetic or a question",
            "Degraded but usable",
            "Blocking one workflow",
            "Complete outage",
        ]);

        Triage {
            client,
            questions: Arc::new(q),
            is_urgent,
            is_churn_risk,
            severity,
        }
    }

    async fn assess(&self, ticket: &str) -> Result<(), typesafe::Error> {
        let answers = self
            .client
            .system_one(&self.questions)
            .state(ticket)
            .send()
            .await?;

        println!("  urgent:     {:.0}%", answers[self.is_urgent].noul * 100.0);
        println!(
            "  churn risk: {:.0}%",
            answers[self.is_churn_risk].noul * 100.0
        );
        println!(
            "  severity:   {:.2} / {}",
            answers[self.severity].score,
            answers[self.severity].levels() - 1
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let triage = Triage::new(Client::new()?);

    let tickets = [
        "Our API integration started returning 500 errors on every request \
         about 20 minutes ago, and we can't process any customer orders.",
        "Quick question — where do I find my invoice history?",
        "This is the third outage this quarter. We're evaluating alternatives.",
    ];

    for ticket in tickets {
        println!("\n{}", &ticket[..ticket.len().min(70)]);
        triage.assess(ticket).await?;
    }

    Ok(())
}
