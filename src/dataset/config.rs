//! Dataset definition format — matches
//! <https://github.com/qdrant/vector-db-benchmark/blob/master/datasets/datasets.json>.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A single dataset entry (from `datasets.json` or inline in upload config).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    pub name: String,
    #[serde(rename = "type", default)]
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
            .with_context(|| format!("dataset {:?}: missing `type`", inline.name))?;
        let path = inline
            .path
            .or_else(|| base.and_then(|b| b.path.clone()))
            .with_context(|| format!("dataset {:?}: missing `path`", inline.name))?;
        let link = inline.link.or_else(|| base.and_then(|b| b.link.clone()));
        if link.is_none() && base.is_none() && !std::path::Path::new(&path).exists() {
            bail!(
                "dataset {:?} not found in datasets.json and no download link provided",
                inline.name
            );
        }
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
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetKind {
    H5,
    Tar,
    Sparse,
}

impl DatasetConfig {
    /// Merge a partial inline definition with a registry entry looked up by `name`.
    pub fn resolve(
        inline: Self,
        registry: &HashMap<String, DatasetConfig>,
    ) -> Result<ResolvedDatasetConfig> {
        inline.resolved(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                schema: None,
            },
        );

        let inline = DatasetConfig {
            name: "glove-100-angular".to_string(),
            kind: None,
            path: None,
            link: None,
            vector_size: None,
            distance: None,
            schema: None,
        };

        let resolved = DatasetConfig::resolve(inline, &registry).unwrap();
        assert_eq!(resolved.path, "glove-100-angular/glove-100-angular.hdf5");
        assert_eq!(resolved.vector_size, Some(100));
        assert!(resolved.link.is_some());
    }
}
