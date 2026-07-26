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
    /// Sharded dataset: a numbered family of files read as one row space.
    /// Mutually exclusive with `path` / `link`.
    #[serde(default)]
    pub parts: Option<PartsConfig>,
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

/// A numbered family of files making up one dataset.
///
/// `path` and `link` are templates containing `{i}`, substituted with each
/// part's number. Part row counts are always measured rather than configured —
/// see [`crate::dataset::parts`] for why a "rows per part" setting would be
/// actively wrong.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartsConfig {
    /// Number of parts; indices run `start .. start + count`.
    pub count: usize,
    #[serde(default)]
    pub start: usize,
    /// Path template resolved against the datasets dir, e.g. `laion/img_emb_{i}.npy`.
    pub path: String,
    /// Download template, e.g. `https://host/img_emb_{i}.npy`.
    #[serde(default)]
    pub link: Option<String>,
}

/// Fully-resolved dataset configuration.
#[derive(Debug, Clone)]
pub struct ResolvedDatasetConfig {
    pub name: String,
    pub kind: DatasetKind,
    /// Path of the single dataset file/directory. Empty when `parts` is set.
    pub path: String,
    pub link: Option<String>,
    pub parts: Option<PartsConfig>,
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
        let parts = inline.parts.or_else(|| base.and_then(|b| b.parts.clone()));
        // A sharded dataset locates its files through the `parts` templates, so
        // the single-file `path` is not required (and must not be set).
        let path = match &parts {
            Some(_) => String::new(),
            None => inline
                .path
                .or_else(|| base.and_then(|b| b.path.clone()))
                .with_context(|| format!("dataset {:?}: missing `path`", inline.name))?,
        };
        let link = inline.link.or_else(|| base.and_then(|b| b.link.clone()));
        // Whether the file actually exists is checked by `ensure_downloaded`, which
        // resolves `path` against the datasets dir; checking it here (relative to the
        // cwd) would wrongly reject local datasets under a custom `BFB_DATASETS_DIR`.
        Ok(ResolvedDatasetConfig {
            name: inline.name,
            kind,
            path,
            link,
            parts,
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
        let Some(kind) = self.kind else {
            bail!("dataset {:?} requires `format` ({KINDS})", self.name);
        };

        if let Some(parts) = &self.parts {
            if self.path.is_some() || self.link.is_some() {
                bail!(
                    "dataset {:?} sets both `parts` and `path`/`link`; \
                     a sharded dataset locates its files through `parts.path` / `parts.link`",
                    self.name
                );
            }
            if !matches!(kind, DatasetKind::Npy | DatasetKind::Parquet) {
                bail!(
                    "dataset {:?}: `parts` is only supported for `format: npy` or \
                     `format: parquet`, not {kind:?}",
                    self.name
                );
            }
            if parts.count == 0 {
                bail!(
                    "dataset {:?}: `parts.count` must be greater than 0",
                    self.name
                );
            }
            // Without the placeholder every part resolves to the same file, which
            // would look like a working upload of `count` copies of part one.
            if parts.count > 1 && !parts.path.contains("{i}") {
                bail!(
                    "dataset {:?}: `parts.path` must contain `{{i}}` to distinguish parts",
                    self.name
                );
            }
            if let Some(link) = &parts.link
                && parts.count > 1
                && !link.contains("{i}")
            {
                bail!(
                    "dataset {:?}: `parts.link` must contain `{{i}}` to distinguish parts",
                    self.name
                );
            }
            return Ok(());
        }

        if self.path.is_none() {
            bail!("dataset {:?} requires `path` (or `parts`)", self.name);
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

    fn parts_config(parts: PartsConfig, kind: DatasetKind) -> DatasetConfig {
        DatasetConfig {
            name: "sharded".to_string(),
            kind: Some(kind),
            parts: Some(parts),
            ..Default::default()
        }
    }

    fn template(path: &str, count: usize) -> PartsConfig {
        PartsConfig {
            count,
            start: 0,
            path: path.to_string(),
            link: None,
        }
    }

    #[test]
    fn parts_resolve_without_a_single_file_path() {
        let config = parts_config(template("laion/img_emb_{i}.npy", 410), DatasetKind::Npy);
        config.validate_inline().unwrap();
        let resolved = DatasetConfig::resolve(config, &HashMap::new()).unwrap();
        assert_eq!(resolved.parts.unwrap().count, 410);
        assert!(resolved.path.is_empty());
    }

    /// Without `{i}` every part resolves to the same file — which would look
    /// like a successful upload of `count` copies of part one.
    #[test]
    fn parts_path_must_distinguish_parts() {
        let config = parts_config(template("laion/img_emb.npy", 410), DatasetKind::Npy);
        let err = config.validate_inline().unwrap_err().to_string();
        assert!(err.contains("{i}"), "{err}");

        // A single part needs no placeholder.
        parts_config(template("laion/img_emb.npy", 1), DatasetKind::Npy)
            .validate_inline()
            .unwrap();
    }

    #[test]
    fn parts_and_path_are_mutually_exclusive() {
        let mut config = parts_config(template("p_{i}.npy", 2), DatasetKind::Npy);
        config.path = Some("p.npy".to_string());
        let err = config.validate_inline().unwrap_err().to_string();
        assert!(err.contains("both `parts` and `path`"), "{err}");
    }

    #[test]
    fn parts_are_rejected_for_bundle_formats() {
        let config = parts_config(template("d_{i}", 3), DatasetKind::Tar);
        let err = config.validate_inline().unwrap_err().to_string();
        assert!(err.contains("only supported for"), "{err}");
    }

    #[test]
    fn parts_count_must_be_positive() {
        let config = parts_config(template("p_{i}.npy", 0), DatasetKind::Npy);
        let err = config.validate_inline().unwrap_err().to_string();
        assert!(err.contains("greater than 0"), "{err}");
    }
}
