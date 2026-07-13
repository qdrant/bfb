//! YAML scroll configuration for `bfb scroll --file config.yaml`.
//!
//! Describes only the *shape* of scroll requests (which collection, which
//! payload filters). The *how* (number of requests, limit, parallelism, uri, …)
//! stays on the CLI.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::search::{FilterPayloadConfig, SearchCollectionConfig};

/// Top-level document: `{ collection: { name }, requests: [ … ] }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrollConfig {
    pub collection: SearchCollectionConfig,
    pub requests: Vec<ScrollRequestConfig>,
}

/// One scroll-request template. At benchmark time one is picked at random per
/// request, with fresh random filter values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrollRequestConfig {
    /// Payload conditions, ANDed together. Empty scrolls the whole collection.
    #[serde(default)]
    pub filters: Vec<FilterPayloadConfig>,
}

pub fn load(path: &str) -> Result<ScrollConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read scroll config file {path}"))?;
    let config: ScrollConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse scroll config file {path}"))?;
    config.validate()?;
    Ok(config)
}

impl ScrollConfig {
    pub fn validate(&self) -> Result<()> {
        if self.requests.is_empty() {
            bail!("scroll config must declare at least one entry under `requests`");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_filtered_and_unfiltered_templates() {
        let config: ScrollConfig = serde_yaml::from_str(
            "collection:\n  name: x\nrequests:\n  - filters: []\n  - filters:\n      - name: color\n        type: keyword\n        source: { type: random, cardinality: 100 }\n",
        )
        .unwrap();
        config.validate().unwrap();

        assert_eq!(config.collection.name, "x");
        assert!(config.requests[0].filters.is_empty());
        assert_eq!(config.requests[1].filters[0].name, "color");
    }

    #[test]
    fn rejects_a_config_with_no_requests() {
        let config: ScrollConfig =
            serde_yaml::from_str("collection:\n  name: x\nrequests: []\n").unwrap();
        assert!(config.validate().is_err());
    }
}
