use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use tar::Archive;

use super::config::{DatasetKind, ResolvedDatasetConfig};

/// Is `path` an `http(s)://` URL rather than a local filesystem path?
pub fn is_remote_url(path: &str) -> bool {
    path.starts_with("http://") || path.starts_with("https://")
}

/// Resolve a vector-source path to a local file, downloading it first if it is
/// an `http(s)://` URL. Remote files are cached under `datasets_dir/files/` and
/// keyed by URL, so repeated runs download once.
///
/// Local paths are returned unchanged.
pub fn ensure_local_file(datasets_dir: &Path, path: &str) -> Result<PathBuf> {
    if !is_remote_url(path) {
        return Ok(PathBuf::from(path));
    }

    let target = datasets_dir.join("files").join(cache_key(path));
    if target.exists() {
        return Ok(target);
    }

    println!("Downloading {path}...");
    let tmp = download_to_temp(path)?;
    let parent = target
        .parent()
        .expect("cache path always has a `files` parent");
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    // Copy rather than rename: the temp file lives on the system temp mount,
    // which is often a different filesystem than the datasets dir.
    fs::copy(&tmp, &target)
        .with_context(|| format!("failed to copy download to {}", target.display()))?;
    let _ = fs::remove_file(&tmp);
    Ok(target)
}

/// A filesystem-safe cache name for `url`: its basename prefixed with a hash of
/// the full URL, so two hosts serving `vectors.fbin` don't collide.
fn cache_key(url: &str) -> String {
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    let hash = hasher.finish();

    let basename = url
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("download")
        .split(['?', '#'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("download");

    format!("{hash:016x}-{basename}")
}

/// Ensure the dataset files exist locally, downloading and extracting if needed.
pub fn ensure_downloaded(datasets_dir: &Path, config: &ResolvedDatasetConfig) -> Result<PathBuf> {
    let target = datasets_dir.join(&config.path);
    if target.exists() {
        return Ok(target);
    }

    let link = config.link.as_deref().with_context(|| {
        format!(
            "dataset {:?} is missing at {}",
            config.name,
            target.display()
        )
    })?;

    println!("Downloading dataset {:?} from {link}...", config.name);
    let tmp = download_to_temp(link)?;
    install_download(&tmp, &target, link, config.kind)?;
    let _ = fs::remove_file(&tmp);
    Ok(target)
}

fn download_to_temp(url: &str) -> Result<PathBuf> {
    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .user_agent("Mozilla/5.0")
            .build(),
    );
    let response = agent
        .get(url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;

    let mut reader = response.into_body().into_reader();
    let mut tmp = tempfile::NamedTempFile::new().context("failed to create temp file")?;
    io::copy(&mut reader, tmp.as_file_mut()).context("failed to write download")?;
    // `keep()` disables auto-deletion so the file survives until `ensure_downloaded`
    // installs it and removes it explicitly; otherwise the `TempPath` would drop and
    // unlink the file the moment this function returns.
    let (_, path) = tmp.keep().context("failed to persist temp download")?;
    Ok(path)
}

fn install_download(tmp: &Path, target: &Path, link: &str, kind: DatasetKind) -> Result<()> {
    if link.ends_with(".tgz") || link.ends_with(".tar.gz") {
        fs::create_dir_all(target)
            .with_context(|| format!("failed to create {}", target.display()))?;
        let file = File::open(tmp).with_context(|| format!("failed to open {}", tmp.display()))?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        archive
            .unpack(target)
            .with_context(|| format!("failed to extract archive into {}", target.display()))?;
        return Ok(());
    }

    match kind {
        DatasetKind::H5 => {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            fs::copy(tmp, target)
                .with_context(|| format!("failed to copy download to {}", target.display()))?;
        }
        DatasetKind::Tar | DatasetKind::Sparse => {
            bail!(
                "dataset archive at {link} must end with .tgz or .tar.gz for type {:?}",
                kind
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_remote_urls() {
        assert!(is_remote_url("https://example.com/v.fbin"));
        assert!(is_remote_url("http://example.com/v.fbin"));
        assert!(!is_remote_url("s3://bucket/v.fbin"));
        assert!(!is_remote_url("./v.fbin"));
        assert!(!is_remote_url("/abs/v.fbin"));
    }

    #[test]
    fn local_paths_pass_through_unchanged() {
        let dir = Path::new("/tmp/datasets");
        assert_eq!(
            ensure_local_file(dir, "./vectors.fbin").unwrap(),
            PathBuf::from("./vectors.fbin")
        );
    }

    #[test]
    fn cache_key_keeps_basename_and_disambiguates_hosts() {
        let a = cache_key("https://a.example.com/vectors.fbin");
        let b = cache_key("https://b.example.com/vectors.fbin");
        assert!(a.ends_with("-vectors.fbin"), "{a}");
        assert!(b.ends_with("-vectors.fbin"), "{b}");
        assert_ne!(a, b, "same basename on different hosts must not collide");
        assert_eq!(a, cache_key("https://a.example.com/vectors.fbin"));
    }

    #[test]
    fn cache_key_strips_query_and_fragment() {
        assert!(
            cache_key("https://x.com/vectors.fbin?token=abc").ends_with("-vectors.fbin"),
            "query string must not leak into the cache filename"
        );
    }

    #[test]
    fn cache_key_tolerates_trailing_slash() {
        assert!(cache_key("https://example.com/data/").ends_with("-data"));
    }

    #[test]
    fn downloads_remote_file_once_and_caches_it() {
        let body = b"fbin-payload".to_vec();
        // Only one request is ever served: a second download attempt would hang
        // the accept loop rather than succeed, proving the cache is used.
        let (url, server) = crate::dataset::test_http::serve_once(body.clone(), 1);

        let dir = tempfile::tempdir().unwrap();
        let first = ensure_local_file(dir.path(), &url).unwrap();
        assert!(first.starts_with(dir.path().join("files")));
        assert_eq!(std::fs::read(&first).unwrap(), body);

        // Second call must hit the cache and return the same path.
        let second = ensure_local_file(dir.path(), &url).unwrap();
        assert_eq!(first, second);

        assert_eq!(server.join().unwrap(), 1, "file should be downloaded once");
    }

    #[test]
    fn reports_download_failure_instead_of_panicking() {
        // Nothing is listening on this port.
        let dir = tempfile::tempdir().unwrap();
        let err = ensure_local_file(dir.path(), "http://127.0.0.1:1/vectors.fbin").unwrap_err();
        assert!(err.to_string().contains("failed to download"), "{err}");
    }
}
