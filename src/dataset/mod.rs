mod config;
mod download;
#[cfg(test)]
pub(crate) mod fixtures;
mod payload;
mod reader;
mod readers;
mod registry;
mod sources;
mod upload;

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
}
