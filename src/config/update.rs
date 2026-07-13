//! YAML patch configuration for `bfb update-collection --file config.yaml`.
//!
//! Every field is optional: only what the file declares is sent, so the rest of
//! the collection config is left alone.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::default_collection_name;

/// Top-level document: `{ collection: { name, hnsw, optimizers } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateConfig {
    pub collection: UpdateCollectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateCollectionConfig {
    #[serde(default = "default_collection_name")]
    pub name: String,
    #[serde(default)]
    pub hnsw: Option<HnswPatch>,
    #[serde(default)]
    pub optimizers: Option<OptimizersPatch>,
}

/// HNSW settings to change. `on_disk` is an `Option<bool>` rather than a plain
/// `bool` so that "not declared" stays distinct from "set to false".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswPatch {
    pub m: Option<u64>,
    pub payload_m: Option<u64>,
    pub ef_construct: Option<u64>,
    pub on_disk: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizersPatch {
    pub default_segment_number: Option<u64>,
    pub indexing_threshold: Option<u64>,
    pub memmap_threshold: Option<u64>,
    pub max_segment_size: Option<u64>,
}

impl UpdateConfig {
    pub fn validate(&self) -> Result<()> {
        let hnsw = self.collection.hnsw.clone().unwrap_or_default();
        let optimizers = self.collection.optimizers.clone().unwrap_or_default();

        let declares_nothing = hnsw.m.is_none()
            && hnsw.payload_m.is_none()
            && hnsw.ef_construct.is_none()
            && hnsw.on_disk.is_none()
            && optimizers.default_segment_number.is_none()
            && optimizers.indexing_threshold.is_none()
            && optimizers.memmap_threshold.is_none()
            && optimizers.max_segment_size.is_none();

        if declares_nothing {
            bail!(
                "update config declares no changes: set at least one field under \
                 `hnsw` or `optimizers`"
            );
        }
        Ok(())
    }
}

pub fn load(path: &str) -> Result<UpdateConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read update config file {path}"))?;
    let config: UpdateConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse update config file {path}"))?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_partial_patch() {
        let config: UpdateConfig = serde_yaml::from_str(
            "collection:\n  name: x\n  optimizers:\n    indexing_threshold: 1\n",
        )
        .unwrap();
        config.validate().unwrap();

        assert_eq!(config.collection.name, "x");
        assert_eq!(
            config.collection.optimizers.unwrap().indexing_threshold,
            Some(1)
        );
        // Undeclared sections stay absent rather than defaulting to something
        // that would be sent to the server.
        assert!(config.collection.hnsw.is_none());
    }

    #[test]
    fn rejects_a_patch_that_changes_nothing() {
        let config: UpdateConfig = serde_yaml::from_str("collection:\n  name: x\n").unwrap();
        assert!(config.validate().is_err());
    }
}
