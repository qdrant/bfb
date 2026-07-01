use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use tar::Archive;

use super::config::{DatasetKind, ResolvedDatasetConfig};

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
