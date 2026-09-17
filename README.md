# typesafe-ai-rs

A Rust client for the [TypeSafe AI](https://docs.typesafe.ai) System One API.

TypeSafe's Jev is a *System One* model: instead of generating prose, it evaluates
some **state** against a set of typed **questions** and returns structured answers
with calibrated probabilities — values your code can branch on directly.

```rust
use typesafe::{Client, QuestionSet};

let client = Client::new()?; // reads TYPESAFE_API_KEY

let mut q = QuestionSet::new();
let urgent = q.noul("is_urgent", "Does this convey urgency?");
let dept = q.choice("department", "Which team should handle this?", [
    ("billing",   "Payments, invoicing, refunds"),
    ("technical", "Bugs, outages, integrations"),
    ("sales",     "Pricing, upgrades, new accounts"),
]);
let frustration = q.score("frustration", "How frustrated is the customer?", [
    "Calm", "Frustrated", "Very angry",
]);

let answers = client
    .system_one(&q)
    .state("Help! My payouts have been failing for 3 days.")
    .send()
    .await?;

let is_urgent: f64 = answers[urgent].noul;
let team: &str     = &answers[dept].choice;
let score: f64     = answers[frustration].score;
```

## Typed answers, not stringly-typed lookups

Each question builder returns a key carrying that question's type, and the
response is indexed by those keys. `answers[urgent]` is a `NoulAnswer`,
`answers[dept]` is a `ChoiceAnswer` — no `match`, no `unwrap`, and no way to ask
a Choice question and read a Score.

Indexing cannot fail for a key from the set that produced the response: the
response is validated on arrival, so a missing or mistyped answer surfaces as
`Error::Protocol` at the call site rather than as a panic downstream. A key from
a *different* `QuestionSet` panics rather than quietly hand back a
plausible-looking number from the wrong question; `SystemOne::get` is the
non-panicking form.

Some invariants are enforced at compile time. The API documents that a Score
rubric needs at least two levels, and so does the signature:

```rust
q.score("frustration", "How frustrated?", ["only one level"]);
// error: evaluation panicked: a score question needs at least 2 levels;
//        use score_iter for a rubric built at runtime
```

> **Stricter than the server.** The live endpoint actually accepts a
> single-level rubric and answers `200`, despite its own docs saying otherwise.
> It is not worth sending: the answer is degenerate by construction — score
> `0.0`, confidence `1.0`, every time — so this crate refuses rather than let
> you pay tokens for a foregone conclusion. Use `score_iter` if you need a
> rubric built at runtime; it reports the same rule as a validation error.

## Ask everything at once

Every question in a set is evaluated in one call, in parallel. TypeSafe's docs
put it plainly: *"asking a question you might not need is close to free."* Build
one `QuestionSet`, keep it, and reuse it — keys are `Copy` and stay valid for the
set's lifetime, so the set and its keys belong together in a struct. See
[`examples/fan_out.rs`](examples/fan_out.rs).

## Runtime support

`Client` requires a **tokio** runtime, because `reqwest` does. This is not a
choice this crate makes, and not one it can forward: every async path in reqwest
is wired to tokio, and reqwest exposes no `async-std`, `smol`, or `runtime-*`
feature to pass along.

What *is* forwarded are the transport knobs reqwest actually has:

| Feature | Effect |
|---|---|
| `rustls-tls` *(default)* | TLS via rustls |
| `native-tls` | TLS via the platform library |
| `gzip`, `brotli`, `deflate`, `zstd` | response decompression |
| `socks` | SOCKS5 proxies |
| `http2` *(default)*, `http3` | HTTP versions |
| `blocking` | the synchronous `blocking::Client` |

On a non-tokio runtime, either enable `blocking` — `reqwest::blocking` runs its
own tokio reactor on a background thread — or wrap calls in
[`async_compat::Compat`](https://docs.rs/async-compat).

> `http3` additionally needs `RUSTFLAGS="--cfg reqwest_unstable"`, which is
> reqwest's requirement. Because of it, `--all-features` does not build.

For transport settings this crate does not expose (proxies, custom certificates,
pool tuning), hand `ClientBuilder::http_client` a pre-configured `reqwest::Client`.

## Configuration

Explicit values beat environment variables, which beat defaults. Blank and
whitespace-only environment values are ignored, matching the JavaScript SDK.

| Setting | Environment variable | Default |
|---|---|---|
| API key | `TYPESAFE_API_KEY` | *required* |
| Base URL | `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` |
| Model | `TYPESAFE_DEFAULT_MODEL` | `jev-latest` |
| Timeout | — | 10s **per attempt** |

There is no total budget across retries, matching the other SDKs: a request that
retries twice can take three times the timeout, plus backoff.

Logging goes through [`tracing`](https://docs.rs/tracing) rather than the
`TYPESAFE_LOG_LEVEL` variable the other SDKs use, so your application controls
filtering. The API key is marked as a sensitive header and is redacted from
`Debug` output.

On macOS, keeping the key in the login keychain rather than a dotfile:

```sh
security add-generic-password -s typesafe-ai -a TYPESAFE_API_KEY -w '<key>' -U
export TYPESAFE_API_KEY=$(security find-generic-password -s typesafe-ai -a TYPESAFE_API_KEY -w)
```

## Errors

`Error::Api` carries the status, the server's message, and its `error_type`,
with predicates for the documented cases. Two things the API does that the docs
do not mention, both handled:

- Error bodies nest the message one level: `{"detail": {"error_type": "...",
  "message": "..."}}`. The message is unwrapped, so you get the prose rather
  than a stringified blob.
- A **missing** `Authorization` header comes back as `403`, not `401`. So
  `is_auth()` covers `401` *and* `403` with an `authentication_error`, and
  `is_permission_denied()` excludes that case — otherwise the single most
  common misconfiguration would not read as an auth failure.

## Retries

`RetryPolicy` defaults to what the official SDKs do: two retries, exponential
backoff from 500ms doubling to a 5s ceiling with 25% jitter, retrying `408`,
`429` and `5xx` plus connection errors and timeouts, honouring `Retry-After` and
`retry-after-ms` up to a minute. TypeSafe warns that rate limits move without
notice while they scale, so leaving this on is worthwhile.

```rust
let policy = RetryPolicy::default()
    .with_max_retries(1)
    .with_backoff_max(Duration::from_secs(1));
```

## Coverage

| Endpoint | Method |
|---|---|
| `POST /v1/systemone` | `client.system_one(&questions)` |
| `GET /v1/models` | `client.models()` |

That is the whole public API surface TypeSafe documents.

## Testing

```sh
cargo test --features blocking
cargo clippy --all-targets --features blocking -- -D warnings
```

Integration tests run against a canned-response server in `tests/support`,
hand-rolled rather than pulled from a mocking crate because the retry tests need
an exact response *sequence* (429 then 200) to prove retries actually retry.

There is also a live suite, ignored by default so `cargo test` stays hermetic
and free:

```sh
export TYPESAFE_API_KEY=$(security find-generic-password -s typesafe-ai -a TYPESAFE_API_KEY -w)
cargo test --test live -- --ignored --test-threads=1
```

It makes at most one billable call per test against deliberately short states.
TypeSafe bills input tokens only, at $0.042 per million, so a full run costs a
small fraction of a cent.

The examples also talk to the live API:

```sh
cargo run --example triage
cargo run --example fan_out
cargo run --example blocking --features blocking
```

## License

MIT OR Apache-2.0
