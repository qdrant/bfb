use std::path::Path;

use anyhow::Result;

use crate::config::UploadConfig;

use super::config::DatasetConfig;
use super::reader::DatasetReader;

/// Collect all dataset configs referenced by an upload config.
pub fn collect_dataset_configs(config: &UploadConfig) -> Vec<DatasetConfig> {
    let mut out = Vec::new();
    for vector in &config.collection.vectors {
        if let crate::config::VectorSource::Dataset { dataset } = &vector.source {
            out.push(dataset.clone());
        }
    }
    for sparse in &config.collection.sparse_vectors {
        if let Some(dataset) = &sparse.source.dataset {
            out.push(dataset.clone());
        }
    }
    for field in &config.collection.fields {
        if let Some(dataset) = field.source.as_ref().and_then(|s| s.dataset.as_ref()) {
            out.push(dataset.clone());
        }
    }
    if let Some(dataset) = config
        .collection
        .payload
        .source
        .as_ref()
        .and_then(|s| s.dataset.as_ref())
    {
        out.push(dataset.clone());
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
pub fn resolve_num_vectors(
    requested: Option<usize>,
    config: &UploadConfig,
    datasets_dir: &Path,
) -> Result<usize> {
    let limit = dataset_point_limit(config, datasets_dir)?;
    Ok(match (requested, limit) {
        (Some(n), Some(limit)) => n.min(limit),
        (Some(n), None) => n,
        (None, Some(limit)) => limit,
        (None, None) => 100_000,
    })
}
