use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use tar::Archive;
use tempfile::NamedTempFile;

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

    let cache_dir = datasets_dir.join("files");
    let target = cache_dir.join(cache_key(path));
    if target.exists() {
        return Ok(target);
    }

    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create {}", cache_dir.display()))?;

    println!("Downloading {path}...");
    // Download into the cache dir, then rename into place. The rename is atomic (same
    // filesystem), so an interrupted or failed download never leaves a truncated file
    // at `target` that later runs would happily reuse.
    let tmp = download_to_temp(path, &cache_dir)?;
    tmp.persist(&target)
        .with_context(|| format!("failed to install download at {}", target.display()))?;
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

    let staging_dir = target.parent().unwrap_or(datasets_dir);
    fs::create_dir_all(staging_dir)
        .with_context(|| format!("failed to create {}", staging_dir.display()))?;

    println!("Downloading dataset {:?} from {link}...", config.name);
    // Stage next to the target so installing it is a same-filesystem rename.
    let tmp = download_to_temp(link, staging_dir)?;
    install_download(tmp, &target, link, config.kind)?;
    Ok(target)
}

/// Download `url` to `target`, creating parent directories as needed.
///
/// Staged next to the target and renamed into place, so an interrupted download
/// never leaves a truncated file that a later run would happily reuse.
pub fn download_file_to(url: &str, target: &Path) -> Result<()> {
    let parent = target.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    println!("Downloading {url}...");
    let tmp = download_to_temp(url, parent)?;
    tmp.persist(target)
        .with_context(|| format!("failed to install download at {}", target.display()))?;
    Ok(())
}

/// A byte range fetched from a remote file, plus the file's full length as
/// reported in `Content-Range`.
pub struct RangeResponse {
    pub body: Vec<u8>,
    pub total_len: u64,
}

/// Fetch `range` (a `Range:` header value such as `bytes=0-511` or `bytes=-65536`)
/// from `url`.
///
/// This is what lets a remote part be *sized* without being downloaded: both the
/// `.npy` header and the parquet footer live at a known end of the file. A server
/// that ignores the range is rejected rather than silently streaming gigabytes.
pub fn fetch_range(url: &str, range: &str) -> Result<RangeResponse> {
    let response = agent()
        .get(url)
        .header("Range", range)
        .call()
        .with_context(|| format!("failed to fetch {range} of {url}"))?;

    if response.status() != 206 {
        bail!(
            "{url} answered {} to a `Range: {range}` request; \
             ranged requests are required to size dataset parts without downloading them",
            response.status()
        );
    }

    // `Content-Range: bytes <start>-<end>/<total>`
    let total_len = response
        .headers()
        .get("content-range")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit_once('/'))
        .and_then(|(_, total)| total.trim().parse::<u64>().ok())
        .with_context(|| format!("{url} returned a Content-Range without a total length"))?;

    let mut body = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut body)
        .with_context(|| format!("failed to read {range} of {url}"))?;

    Ok(RangeResponse { body, total_len })
}

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .user_agent("Mozilla/5.0")
            .build(),
    )
}

/// Download `url` into a temporary file inside `dir`. The file is deleted when the
/// returned handle is dropped, so an aborted download leaves nothing behind.
fn download_to_temp(url: &str, dir: &Path) -> Result<NamedTempFile> {
    let agent = agent();
    let response = agent
        .get(url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;

    let mut reader = response.into_body().into_reader();
    let mut tmp = NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create temp file in {}", dir.display()))?;
    io::copy(&mut reader, tmp.as_file_mut()).context("failed to write download")?;
    Ok(tmp)
}

fn install_download(
    tmp: NamedTempFile,
    target: &Path,
    link: &str,
    kind: DatasetKind,
) -> Result<()> {
    if link.ends_with(".tgz") || link.ends_with(".tar.gz") {
        // Extract into a staging dir and rename it into place, so a failed extraction
        // does not leave a half-populated dataset dir that later runs treat as complete.
        let parent = target.parent().unwrap_or(Path::new("."));
        let staging = tempfile::TempDir::new_in(parent)
            .with_context(|| format!("failed to create staging dir in {}", parent.display()))?;
        let file = File::open(tmp.path())
            .with_context(|| format!("failed to open {}", tmp.path().display()))?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        archive
            .unpack(staging.path())
            .with_context(|| format!("failed to extract archive into {}", target.display()))?;
        let staged = staging.keep();
        fs::rename(&staged, target).with_context(|| {
            format!(
                "failed to move extracted archive from {} to {}",
                staged.display(),
                target.display()
            )
        })?;
        return Ok(());
    }

    if kind.is_single_file() {
        tmp.persist(target)
            .with_context(|| format!("failed to install download at {}", target.display()))?;
        return Ok(());
    }
    bail!("dataset archive at {link} must end with .tgz or .tar.gz for format {kind:?}")
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
        // Only one request is served; if the cache is not used, the second call will fail.
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

    #[test]
    fn failed_download_leaves_no_cache_entry() {
        let dir = tempfile::tempdir().unwrap();
        let url = "http://127.0.0.1:1/vectors.fbin";
        assert!(ensure_local_file(dir.path(), url).is_err());

        // A partial download must not be persisted under its cache key, otherwise every
        // later run would short-circuit on the truncated file.
        let cached: Vec<_> = fs::read_dir(dir.path().join("files"))
            .map(|entries| entries.map(|e| e.unwrap().path()).collect())
            .unwrap_or_default();
        assert!(cached.is_empty(), "cache dir must be empty, got {cached:?}");
    }
}
