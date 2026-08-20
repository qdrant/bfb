mod conditions;
mod config;
mod download;
#[cfg(test)]
pub(crate) mod fixtures;
mod parts;
mod payload;
mod reader;
mod readers;
mod registry;
mod sources;
mod upload;

pub use conditions::parse as parse_query_conditions;
pub use config::DatasetConfig;
pub use download::{ensure_local_file, is_remote_url};
pub use reader::DatasetReader;
pub use sources::UploadDatasetSources;
pub use upload::resolve_num_vectors;

/// Default directory for downloaded datasets (matches vector-db-benchmark layout).
pub fn default_datasets_dir() -> std::path::PathBuf {
    std::env::var_os("BFB_DATASETS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("datasets"))
}

/// A throwaway HTTP server for exercising the download path in tests.
#[cfg(test)]
pub(crate) mod test_http {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// Serve `body` over HTTP to exactly `hits` clients, then stop. Returns the
    /// URL and a handle yielding how many requests were actually served, so a
    /// caller can assert that a cached file was not re-fetched.
    pub(crate) fn serve_once(body: Vec<u8>, hits: usize) -> (String, JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/vectors.fbin", listener.local_addr().unwrap());

        let handle = std::thread::spawn(move || {
            let mut served = 0;
            for _ in 0..hits {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
                served += 1;
            }
            served
        });

        (url, handle)
    }

    /// What a [`serve_ranges`] server actually did.
    pub(crate) struct ServeStats {
        pub requests: usize,
        /// Total body bytes written back, so a test can assert that sizing a
        /// dataset did *not* amount to downloading it.
        pub bytes_served: usize,
    }

    /// Serve `files` (by path, e.g. `"p_0.npy"`) over HTTP with `Range:`
    /// support, for exactly `max_requests` requests. Returns the base URL and a
    /// handle yielding the request/byte counts.
    pub(crate) fn serve_ranges(
        files: Vec<(String, Vec<u8>)>,
        max_requests: usize,
    ) -> (String, JoinHandle<ServeStats>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());

        let handle = std::thread::spawn(move || {
            let mut stats = ServeStats {
                requests: 0,
                bytes_served: 0,
            };
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 2048];
                let read = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..read]).to_string();

                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/")
                    .trim_start_matches('/')
                    .to_string();
                let Some((_, body)) = files.iter().find(|(name, _)| *name == path) else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    stats.requests += 1;
                    continue;
                };

                let range = request
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("range:"))
                    .and_then(|line| {
                        line.split_once('=')
                            .map(|(_, spec)| spec.trim().to_string())
                    });

                let total = body.len();
                let (status, start, end) = match range.as_deref().map(parse_range) {
                    // `bytes=-N`: the last N bytes.
                    Some((None, Some(suffix))) => {
                        (206, total.saturating_sub(suffix), total.saturating_sub(1))
                    }
                    // `bytes=A-B` (B optional, and clamped to the real end).
                    Some((Some(from), to)) => (206, from, to.unwrap_or(total - 1).min(total - 1)),
                    _ => (200, 0, total.saturating_sub(1)),
                };
                let slice = &body[start..=end.min(total - 1)];

                let mut header = format!(
                    "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n",
                    if status == 206 {
                        "Partial Content"
                    } else {
                        "OK"
                    },
                    slice.len()
                );
                if status == 206 {
                    header.push_str(&format!(
                        "Content-Range: bytes {start}-{}/{total}\r\n",
                        start + slice.len() - 1
                    ));
                }
                header.push_str("Connection: close\r\n\r\n");

                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(slice);
                let _ = stream.flush();
                stats.requests += 1;
                stats.bytes_served += slice.len();
            }
            stats
        });

        (base, handle)
    }

    /// Parse a `Range` header spec into `(start, end)`; a bare `-N` suffix
    /// range comes back as `(None, Some(N))`.
    fn parse_range(spec: &str) -> (Option<usize>, Option<usize>) {
        let (from, to) = spec.split_once('-').unwrap_or((spec, ""));
        (from.trim().parse().ok(), to.trim().parse().ok())
    }
}
