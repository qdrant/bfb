use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::config::DatasetConfig;

/// Load an optional `datasets.json` registry from `datasets_dir`.
///
/// Returns an empty map when the file is absent so upload configs can specify
/// datasets fully inline.
pub fn load_registry(datasets_dir: &Path) -> Result<HashMap<String, DatasetConfig>> {
    let path = datasets_dir.join("datasets.json");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let entries: Vec<DatasetConfig> = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(entries
        .into_iter()
        .map(|entry| (entry.name.clone(), entry))
        .collect())
}
