use std::path::Path;

use anyhow::{Result, bail};

use crate::config::UploadConfig;

use super::config::DatasetConfig;
use super::reader::DatasetReader;

/// Collect all dataset configs referenced by an upload config.
pub fn collect_dataset_configs(config: &UploadConfig) -> Vec<DatasetConfig> {
    let mut out = Vec::new();
    for vector in &config.collection.vectors {
        if let crate::config::VectorSource::Dataset { dataset } = &vector.source {
            out.push((**dataset).clone());
        }
    }
    for sparse in &config.collection.sparse_vectors {
        if let Some(dataset) = &sparse.source.dataset {
            out.push((**dataset).clone());
        }
    }
    for field in &config.collection.fields {
        if let Some(dataset) = field.source.as_ref().and_then(|s| s.dataset.as_ref()) {
            out.push((**dataset).clone());
        }
    }
    if let Some(dataset) = config
        .collection
        .payload
        .source
        .as_ref()
        .and_then(|s| s.dataset.as_ref())
    {
        out.push((**dataset).clone());
    }
    out
}

/// Minimum point count across all dataset sources, if any are configured.
pub fn dataset_point_limit(config: &UploadConfig, datasets_dir: &Path) -> Result<Option<usize>> {
    let configs = collect_dataset_configs(config);
    if configs.is_empty() {
        return Ok(None);
    }
    let mut min_points = usize::MAX;
    for dataset_config in configs {
        let reader = DatasetReader::open(datasets_dir, &dataset_config)?;
        min_points = min_points.min(reader.num_points);
    }
    Ok(Some(min_points))
}

/// Resolve the number of points to upload.
///
/// When dataset sources are present and `-n` is omitted, the full dataset (up
/// to the smallest source) is uploaded. When `-n` is set, it is capped by that
/// limit. Without dataset sources the legacy default of 100_000 applies.
///
/// A point's id doubles as its dataset row, so `--offset` skips that many rows
/// too — which is what makes it a resume switch for a long upload. The rows
/// left to read are therefore counted from `offset`, not from zero.
pub fn resolve_num_vectors(
    requested: Option<usize>,
    offset: usize,
    config: &UploadConfig,
    datasets_dir: &Path,
) -> Result<usize> {
    let limit = dataset_point_limit(config, datasets_dir)?;
    if let Some(limit) = limit
        && offset >= limit
    {
        bail!(
            "--offset {offset} starts past the end of the dataset ({limit} rows); \
             nothing would be uploaded"
        );
    }
    Ok(match (requested, limit) {
        (Some(n), Some(limit)) => n.min(limit - offset),
        (Some(n), None) => n,
        (None, Some(limit)) => limit - offset,
        (None, None) => 100_000,
    })
}

#[cfg(test)]
mod tests {
    use crate::config::UploadConfig;
    use crate::dataset::fixtures::make_ramp_npy;

    fn dataset_config(dir: &std::path::Path) -> UploadConfig {
        std::fs::write(dir.join("v.npy"), make_ramp_npy(0, 10, 4)).unwrap();
        serde_yaml::from_str(
            "
collection:
  vectors:
    - size: 4
      source: { type: dataset, name: v, format: npy, path: v.npy }
",
        )
        .unwrap()
    }

    /// `--offset` skips dataset rows as well as ids, so the rows *remaining*
    /// are what bound the run — otherwise resuming a part-way upload would read
    /// past the end of the corpus.
    #[test]
    fn offset_reduces_the_rows_left_to_upload() {
        let dir = tempfile::tempdir().unwrap();
        let config = dataset_config(dir.path());

        let all = super::resolve_num_vectors(None, 0, &config, dir.path()).unwrap();
        assert_eq!(all, 10);

        let resumed = super::resolve_num_vectors(None, 6, &config, dir.path()).unwrap();
        assert_eq!(resumed, 4, "only rows 6..10 are left");

        let capped = super::resolve_num_vectors(Some(100), 6, &config, dir.path()).unwrap();
        assert_eq!(capped, 4, "-n cannot exceed what remains");

        let under = super::resolve_num_vectors(Some(2), 6, &config, dir.path()).unwrap();
        assert_eq!(under, 2);
    }

    #[test]
    fn offset_past_the_end_is_an_error_rather_than_an_empty_run() {
        let dir = tempfile::tempdir().unwrap();
        let config = dataset_config(dir.path());

        let err = super::resolve_num_vectors(None, 10, &config, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("past the end"), "{err}");
    }

    /// Without dataset sources there is nothing to bound, and `--offset` only
    /// shifts ids — the legacy behaviour must be untouched.
    #[test]
    fn offset_does_not_bound_generated_data() {
        let dir = tempfile::tempdir().unwrap();
        let config: UploadConfig =
            serde_yaml::from_str("collection:\n  vectors:\n    - size: 4\n").unwrap();

        assert_eq!(
            super::resolve_num_vectors(Some(5), 1_000_000, &config, dir.path()).unwrap(),
            5
        );
        assert_eq!(
            super::resolve_num_vectors(None, 1_000_000, &config, dir.path()).unwrap(),
            100_000
        );
    }
}
