//! A tiny canned-response HTTP server for integration tests.
//!
//! Hand-rolled rather than pulled from a mocking crate because these tests
//! need an exact response *sequence* — 429 then 200, to prove retries actually
//! retry — which matcher-based mock libraries express awkwardly at best. It
//! also lets the tests assert on the precise request bytes that went out.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One response the server will hand out.
#[derive(Debug, Clone)]
pub struct Canned {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Canned {
    pub fn json(status: u16, body: &str) -> Self {
        Canned {
            status,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body: body.to_owned(),
        }
    }

    pub fn ok(body: &str) -> Self {
        Canned::json(200, body)
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_owned(), value.to_owned()));
        self
    }
}

/// A request the server received.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("request body was not JSON")
    }
}

pub struct MockServer {
    pub url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    port: u16,
    handle: Option<JoinHandle<()>>,
}

impl MockServer {
    /// Serve `responses` in order. Once exhausted, the last one repeats, so a
    /// test that only cares about the happy path can pass a single response.
    pub fn start(responses: Vec<Canned>) -> Self {
        assert!(!responses.is_empty(), "need at least one canned response");

        let listener = TcpListener::bind("127.0.0.1:0").expect("could not bind a test port");
        let port = listener.local_addr().unwrap().port();

        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let handle = {
            let requests = Arc::clone(&requests);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut served = 0usize;
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };

                    match handle_connection(stream, &responses, served) {
                        Some(recorded) => {
                            requests.lock().unwrap().push(recorded);
                            served += 1;
                        }
                        None => continue,
                    }
                }
            })
        };

        MockServer {
            url: format!("http://127.0.0.1:{port}"),
            requests,
            stop,
            port,
            handle: Some(handle),
        }
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    /// How many requests have been received.
    pub fn hits(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock the accept loop so the thread can notice the stop flag.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Read one request, write the canned response, close the socket.
///
/// `Connection: close` on every response keeps this honest: no keep-alive
/// bookkeeping, and each retry shows up as its own connection.
fn handle_connection(
    mut stream: TcpStream,
    responses: &[Canned],
    served: usize,
) -> Option<Recorded> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }

    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_owned();
    let path = parts.next()?.to_owned();

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_owned();
            let value = value.trim().to_owned();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((name, value));
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }

    // Past the end of the list, the last response repeats.
    let canned = responses.get(served).unwrap_or_else(|| {
        responses
            .last()
            .expect("start() guarantees a non-empty list")
    });

    let mut response = format!(
        "HTTP/1.1 {} {}\r\n",
        canned.status,
        reason_phrase(canned.status)
    );
    for (name, value) in &canned.headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str(&format!("content-length: {}\r\n", canned.body.len()));
    response.push_str("connection: close\r\n\r\n");
    response.push_str(&canned.body);

    stream.write_all(response.as_bytes()).ok()?;
    stream.flush().ok()?;
    let _ = stream.shutdown(Shutdown::Write);

    Some(Recorded {
        method,
        path,
        headers,
        body,
    })
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        529 => "Overloaded",
        _ => "Status",
    }
}

/// The response body from the quickstart in TypeSafe's docs.
pub const QUICKSTART_RESPONSE: &str = r#"{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.84, "technical": 0.159, "sales": 0.001},
      "confidence": 0.596
    },
    "frustration": {
      "type": "score",
      "score": 1.035,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "confidence": 0.842
    },
    "is_urgent": {"type": "noul", "noul": 0.999}
  },
  "usage": {"input_tokens": 312, "output_tokens": 48}
}"#;
