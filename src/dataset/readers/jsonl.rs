use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

/// Line-oriented JSON store with byte-offset index for random access.
pub struct JsonlStore {
    path: PathBuf,
    offsets: Vec<u64>,
}

impl JsonlStore {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut offsets = Vec::new();
        let mut offset = 0u64;
        let mut line = String::new();
        loop {
            offsets.push(offset);
            let read = reader.read_line(&mut line)?;
            if read == 0 {
                offsets.pop();
                break;
            }
            offset += read as u64;
            line.clear();
        }
        Ok(JsonlStore {
            path: path.to_path_buf(),
            offsets,
        })
    }

    pub fn value_at(&self, idx: usize) -> Result<Option<Value>> {
        let Some(&byte_offset) = self.offsets.get(idx) else {
            return Ok(None);
        };
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(byte_offset))?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            return Ok(None);
        }
        let value: Value = serde_json::from_str(line.trim())
            .with_context(|| format!("failed to parse payloads.jsonl line {idx}"))?;
        Ok(Some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_store_random_access() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        std::fs::write(&path, "{\"a\":1}\n{\"a\":2}\n").unwrap();
        let store = JsonlStore::open(&path).unwrap();
        assert_eq!(store.value_at(1).unwrap().unwrap()["a"], 2);
    }
}
