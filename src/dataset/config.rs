//! Dataset definition format — compatible with
//! <https://github.com/qdrant/vector-db-benchmark/blob/master/datasets/datasets.json>,
//! but intended to be specified inline in upload config source definitions.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A single dataset entry (inline in upload config, or from an optional local
/// `datasets.json` registry).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    pub name: String,
    /// Dataset storage format. Serialized as `format` in upload configs (the
    /// source already uses `type: dataset`). `type` is accepted as an alias
    /// for compatibility with vector-db-benchmark `datasets.json`.
    #[serde(rename = "format", alias = "type", default)]
    pub kind: Option<DatasetKind>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub link: Option<String>,
    #[serde(default)]
    pub vector_size: Option<u64>,
    #[serde(default)]
    pub distance: Option<String>,
    #[serde(default)]
    pub schema: Option<HashMap<String, String>>,
    /// `parquet` only: columns to keep. `None` ⇒ every column.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// `parquet` only: columns to drop (applied after `columns`).
    #[serde(default)]
    pub exclude: Vec<String>,
    /// `parquet` only: value substituted for nulls and non-finite floats.
    /// Omitted by default, which leaves the payload field absent.
    #[serde(default)]
    pub fill_null: Option<serde_json::Value>,
}

impl DatasetConfig {
    /// Resolved config with required fields populated.
    pub fn resolved(
        self,
        registry: &HashMap<String, DatasetConfig>,
    ) -> Result<ResolvedDatasetConfig> {
        ResolvedDatasetConfig::from_inline(self, registry)
    }
}

/// Fully-resolved dataset configuration.
#[derive(Debug, Clone)]
pub struct ResolvedDatasetConfig {
    pub name: String,
    pub kind: DatasetKind,
    pub path: String,
    pub link: Option<String>,
    #[allow(dead_code)]
    pub vector_size: Option<u64>,
    #[allow(dead_code)]
    pub distance: Option<String>,
    #[allow(dead_code)]
    pub schema: Option<HashMap<String, String>>,
    pub columns: Option<Vec<String>>,
    pub exclude: Vec<String>,
    pub fill_null: Option<serde_json::Value>,
}

impl ResolvedDatasetConfig {
    fn from_inline(
        inline: DatasetConfig,
        registry: &HashMap<String, DatasetConfig>,
    ) -> Result<Self> {
        let base = registry.get(&inline.name);
        let kind = inline
            .kind
            .or_else(|| base.and_then(|b| b.kind))
            .with_context(|| format!("dataset {:?}: missing `format` ({KINDS})", inline.name))?;
        let path = inline
            .path
            .or_else(|| base.and_then(|b| b.path.clone()))
            .with_context(|| format!("dataset {:?}: missing `path`", inline.name))?;
        let link = inline.link.or_else(|| base.and_then(|b| b.link.clone()));
        // Whether the file actually exists is checked by `ensure_downloaded`, which
        // resolves `path` against the datasets dir; checking it here (relative to the
        // cwd) would wrongly reject local datasets under a custom `BFB_DATASETS_DIR`.
        Ok(ResolvedDatasetConfig {
            name: inline.name,
            kind,
            path,
            link,
            vector_size: inline
                .vector_size
                .or_else(|| base.and_then(|b| b.vector_size)),
            distance: inline
                .distance
                .or_else(|| base.and_then(|b| b.distance.clone())),
            schema: inline
                .schema
                .or_else(|| base.and_then(|b| b.schema.clone())),
            columns: inline
                .columns
                .or_else(|| base.and_then(|b| b.columns.clone())),
            exclude: if inline.exclude.is_empty() {
                base.map(|b| b.exclude.clone()).unwrap_or_default()
            } else {
                inline.exclude
            },
            fill_null: inline
                .fill_null
                .or_else(|| base.and_then(|b| b.fill_null.clone())),
        })
    }
}

/// Formats accepted by `format:`, for error messages.
const KINDS: &str = "h5, tar, sparse, npy, parquet";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetKind {
    H5,
    Tar,
    Sparse,
    /// A standalone 2-D float `.npy` array: dense vectors, no payloads.
    Npy,
    /// A parquet file of payload rows: no vectors.
    Parquet,
}

impl DatasetKind {
    /// Is the dataset a single file, rather than a directory of extracted
    /// files? Single-file datasets are installed as-is on download.
    pub fn is_single_file(self) -> bool {
        matches!(
            self,
            DatasetKind::H5 | DatasetKind::Npy | DatasetKind::Parquet
        )
    }
}

impl DatasetConfig {
    /// Merge a partial inline definition with an optional registry entry looked up by `name`.
    pub fn resolve(
        inline: Self,
        registry: &HashMap<String, DatasetConfig>,
    ) -> Result<ResolvedDatasetConfig> {
        inline.resolved(registry)
    }

    /// Validate fields required when no registry entry supplies missing values.
    pub fn validate_inline(&self) -> Result<()> {
        if self.name.is_empty() {
            bail!("dataset source requires `name`");
        }
        if self.kind.is_none() {
            bail!("dataset {:?} requires `format` ({KINDS})", self.name);
        }
        if self.path.is_none() {
            bail!("dataset {:?} requires `path`", self.name);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_fully_inline_without_registry() {
        let inline = DatasetConfig {
            name: "glove-25-angular".to_string(),
            kind: Some(DatasetKind::H5),
            path: Some("glove-25-angular/glove-25-angular.hdf5".to_string()),
            link: Some("http://ann-benchmarks.com/glove-25-angular.hdf5".to_string()),
            vector_size: Some(25),
            distance: Some("cosine".to_string()),
            ..Default::default()
        };

        let resolved = DatasetConfig::resolve(inline, &HashMap::new()).unwrap();
        assert_eq!(resolved.path, "glove-25-angular/glove-25-angular.hdf5");
        assert_eq!(resolved.vector_size, Some(25));
        assert_eq!(resolved.kind, DatasetKind::H5);
    }

    #[test]
    fn merges_registry_entry() {
        let mut registry = HashMap::new();
        registry.insert(
            "glove-100-angular".to_string(),
            DatasetConfig {
                name: "glove-100-angular".to_string(),
                kind: Some(DatasetKind::H5),
                path: Some("glove-100-angular/glove-100-angular.hdf5".to_string()),
                link: Some("http://example.com/glove.hdf5".to_string()),
                vector_size: Some(100),
                distance: Some("cosine".to_string()),
                ..Default::default()
            },
        );

        let inline = DatasetConfig {
            name: "glove-100-angular".to_string(),
            ..Default::default()
        };

        let resolved = DatasetConfig::resolve(inline, &registry).unwrap();
        assert_eq!(resolved.path, "glove-100-angular/glove-100-angular.hdf5");
        assert_eq!(resolved.vector_size, Some(100));
        assert!(resolved.link.is_some());
    }
}
