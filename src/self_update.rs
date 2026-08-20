//! `bfb self-update`: replace the running binary with a published GitHub release.
//!
//! Release assets are produced by `.github/workflows/release.yml` and named
//! `bfb-<target>` (+ `.sha256`), where `<target>` is one of the
//! [`SUPPORTED_TARGETS`]. Each asset is the bare executable, so installing by
//! hand is `curl -o` + `chmod +x`.
//!
//! The swap is a `rename(2)` of a fully written temp file over the current
//! executable, so it is atomic and safe to do while this very process runs.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// GitHub repository releases are fetched from.
pub const REPO: &str = "qdrant/bfb";

/// Current binary version, as baked in by Cargo.
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Target triples the release workflow publishes binaries for.
pub const SUPPORTED_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
];

#[derive(clap::Args, Debug, Clone)]
pub struct SelfUpdateArgs {
    /// Only report whether a newer release is available; do not install it.
    #[clap(long)]
    pub check: bool,

    /// Install this release tag (e.g. `v0.2.0`) instead of the latest one.
    #[clap(long)]
    pub tag: Option<String>,

    /// Reinstall even when the selected release matches the running version.
    #[clap(long)]
    pub force: bool,
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

pub fn run(args: &SelfUpdateArgs) -> Result<()> {
    let release = fetch_release(args.tag.as_deref())?;
    let release_version = release.tag_name.trim_start_matches('v');

    if release_version == CURRENT_VERSION && !args.force {
        println!(
            "bfb {CURRENT_VERSION} is already the selected release ({})",
            release.tag_name
        );
        return Ok(());
    }

    println!("Current version: {CURRENT_VERSION}");
    println!(
        "Release version: {} ({})",
        release_version, release.tag_name
    );
    if args.check {
        println!("Run `bfb self-update` to install it.");
        return Ok(());
    }

    let target = current_target()?;
    let asset_name = format!("bfb-{target}");
    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .with_context(|| {
            format!(
                "release {} has no asset `{asset_name}`; available: {}",
                release.tag_name,
                release
                    .assets
                    .iter()
                    .map(|asset| asset.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let checksum_asset = release
        .assets
        .iter()
        .find(|candidate| candidate.name == format!("{asset_name}.sha256"));

    let exe = std::env::current_exe().context("failed to locate the running executable")?;
    // `current_exe` can be a symlink (e.g. a `~/.local/bin/bfb -> …` install);
    // replace the file it points to rather than the link.
    let exe =
        fs::canonicalize(&exe).with_context(|| format!("failed to resolve {}", exe.display()))?;
    let install_dir = exe
        .parent()
        .with_context(|| format!("{} has no parent directory", exe.display()))?;

    println!("Downloading {}...", asset.browser_download_url);
    let binary = download(&asset.browser_download_url)?;

    if let Some(checksum_asset) = checksum_asset {
        let expected = download(&checksum_asset.browser_download_url)?;
        verify_sha256(&binary, &expected, &asset_name)?;
    } else {
        println!("Warning: release has no `{asset_name}.sha256`; skipping checksum verification");
    }

    // Write next to the target so the final rename stays on one filesystem, and
    // never truncate the running executable in place (that crashes the process).
    let staged = stage_binary(install_dir, &binary).with_context(|| {
        format!(
            "failed to write the new binary into {} (try re-running with `sudo` if the \
             directory is not writable)",
            install_dir.display()
        )
    })?;
    fs::rename(&staged, &exe).map_err(|err| {
        let _ = fs::remove_file(&staged);
        anyhow::anyhow!(
            "failed to replace {} (try re-running with `sudo` if the file is not writable): {err}",
            exe.display()
        )
    })?;

    println!(
        "Updated {} to bfb {release_version} ({}).",
        exe.display(),
        release.tag_name
    );
    Ok(())
}

/// Map the running platform onto a published release target. Linux always maps
/// to the static musl build, so a `-gnu`-compiled dev binary still updates.
fn current_target() -> Result<&'static str> {
    let target = match (std::env::consts::ARCH, std::env::consts::OS) {
        ("x86_64", "linux") => "x86_64-unknown-linux-musl",
        ("aarch64", "linux") => "aarch64-unknown-linux-musl",
        ("x86_64", "macos") => "x86_64-apple-darwin",
        ("aarch64", "macos") => "aarch64-apple-darwin",
        (arch, os) => bail!(
            "no prebuilt bfb binaries are published for {arch}-{os}; supported targets: {}",
            SUPPORTED_TARGETS.join(", ")
        ),
    };
    Ok(target)
}

/// GitHub REST API base. `GITHUB_API_URL` is the conventional override (it is
/// set in GitHub Actions and used for GitHub Enterprise hosts).
fn api_base() -> String {
    std::env::var("GITHUB_API_URL")
        .ok()
        .filter(|url| !url.is_empty())
        .map(|url| url.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://api.github.com".to_string())
}

fn fetch_release(tag: Option<&str>) -> Result<Release> {
    let api = api_base();
    let url = match tag {
        Some(tag) => format!("{api}/repos/{REPO}/releases/tags/{tag}"),
        None => format!("{api}/repos/{REPO}/releases/latest"),
    };
    let body = download(&url).with_context(|| match tag {
        Some(tag) => format!("failed to fetch release `{tag}` of {REPO}"),
        None => format!("failed to fetch the latest release of {REPO}"),
    })?;
    serde_json::from_slice(&body).context("failed to parse the GitHub release response")
}

/// GET `url` and return the body. Follows redirects (release assets redirect to
/// a CDN). Honours `GITHUB_TOKEN` to lift the anonymous API rate limit.
fn download(url: &str) -> Result<Vec<u8>> {
    let mut request = ureq::get(url)
        .header("User-Agent", format!("bfb/{CURRENT_VERSION}"))
        .header(
            "Accept",
            "application/vnd.github+json, application/octet-stream",
        );
    if url.starts_with(&api_base())
        && let Ok(token) = std::env::var("GITHUB_TOKEN")
        && !token.is_empty()
    {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let response = request
        .call()
        .with_context(|| format!("request to {url} failed"))?;
    let mut body = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut body)
        .with_context(|| format!("failed to read response body of {url}"))?;
    Ok(body)
}

/// `expected` is the content of a `sha256sum`-style file: `<hex>  <name>`.
fn verify_sha256(data: &[u8], expected: &[u8], asset_name: &str) -> Result<()> {
    let expected = std::str::from_utf8(expected)
        .context("checksum file is not UTF-8")?
        .split_whitespace()
        .next()
        .context("checksum file is empty")?
        .to_ascii_lowercase();
    let actual = format!("{:x}", Sha256::digest(data));
    if actual != expected {
        bail!("checksum mismatch for {asset_name}: expected {expected}, got {actual}");
    }
    Ok(())
}

/// Write `binary` to a fresh, executable temp file inside `dir`.
fn stage_binary(dir: &Path, binary: &[u8]) -> Result<PathBuf> {
    let staged = dir.join(format!(".bfb-update-{}", std::process::id()));
    let mut file = create_executable(&staged)?;
    let written = (|| -> io::Result<()> {
        file.write_all(binary)?;
        file.sync_all()
    })();
    if let Err(err) = written {
        let _ = fs::remove_file(&staged);
        return Err(err).with_context(|| format!("failed to write {}", staged.display()));
    }
    Ok(staged)
}

#[cfg(unix)]
fn create_executable(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o755)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

#[cfg(not(unix))]
fn create_executable(path: &Path) -> Result<File> {
    File::create(path).with_context(|| format!("failed to create {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_sha256sum_format() {
        let data = b"hello";
        let sum = format!("{:x}", Sha256::digest(data));
        verify_sha256(data, format!("{sum}  bfb\n").as_bytes(), "bfb").unwrap();
        verify_sha256(data, sum.to_uppercase().as_bytes(), "bfb").unwrap();
        assert!(verify_sha256(b"other", sum.as_bytes(), "bfb").is_err());
        assert!(verify_sha256(data, b"", "bfb").is_err());
    }

    #[test]
    fn parses_github_release_json() {
        let json = r#"{"tag_name":"v0.2.0","name":"x","assets":[
            {"name":"bfb-x86_64-unknown-linux-musl","browser_download_url":"https://e/a","size":1}
        ]}"#;
        let release: Release = serde_json::from_str(json).unwrap();
        assert_eq!(release.tag_name, "v0.2.0");
        assert_eq!(release.assets[0].browser_download_url, "https://e/a");
    }

    #[test]
    fn staged_binary_is_executable_and_atomic_to_rename() {
        let dir = tempfile::tempdir().unwrap();
        let staged = stage_binary(dir.path(), b"bin").unwrap();
        assert_eq!(fs::read(&staged).unwrap(), b"bin");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&staged).unwrap().permissions().mode() & 0o777,
                0o755
            );
        }
        let target = dir.path().join("bfb");
        fs::write(&target, b"old").unwrap();
        fs::rename(&staged, &target).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"bin");
    }

    #[test]
    fn current_platform_maps_to_a_published_target() {
        // On unsupported dev platforms this is expected to error instead.
        if let Ok(target) = current_target() {
            assert!(SUPPORTED_TARGETS.contains(&target));
        }
    }
}
