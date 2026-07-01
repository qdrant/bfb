use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::config::DatasetConfig;

/// Load `datasets.json` from `datasets_dir`.
pub fn load_registry(datasets_dir: &Path) -> Result<HashMap<String, DatasetConfig>> {
    let path = datasets_dir.join("datasets.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let entries: Vec<DatasetConfig> =
        serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(entries
        .into_iter()
        .map(|entry| (entry.name.clone(), entry))
        .collect())
}
