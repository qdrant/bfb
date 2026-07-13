//! Memory and on-disk size, from `GET /collections/{c}/memory` and
//! `GET /telemetry`. REST-only, like [`crate::optimizations`].

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Bytes held by one component, on disk and in RAM.
///
/// `cached_bytes` is what the page cache currently holds of the on-disk part;
/// `expected_cache_bytes` is what the component would like cached to serve
/// queries without hitting disk. A gap between them means a cold cache.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub disk_bytes: u64,
    pub ram_bytes: u64,
    pub cached_bytes: u64,
    pub expected_cache_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedUsage {
    pub name: String,
    pub usage: Usage,
}

/// A vector's storage and its index, sized separately.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VectorUsage {
    pub name: String,
    pub storage: Usage,
    pub index: Usage,
}

/// Process-wide allocator stats from `/telemetry`. Not per-collection: on a
/// server with other collections these cover all of them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ProcessMemory {
    pub active_bytes: u64,
    pub allocated_bytes: u64,
    pub metadata_bytes: u64,
    pub resident_bytes: u64,
    pub retained_bytes: u64,
}

/// Where the collection's bytes live, at the moment it was sampled.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryReport {
    pub total: Usage,
    pub vectors: Vec<VectorUsage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sparse_vectors: Vec<VectorUsage>,
    pub payload: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub payload_index: Vec<NamedUsage>,
    /// Everything else the collection stores, e.g. `id_tracker`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub other: BTreeMap<String, Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process: Option<ProcessMemory>,
}

impl MemoryReport {
    pub fn print(&self) {
        println!("--- Memory & disk ---");
        println!(
            "total: {} disk, {} ram, {} cached (of {} expected)",
            mib(self.total.disk_bytes),
            mib(self.total.ram_bytes),
            mib(self.total.cached_bytes),
            mib(self.total.expected_cache_bytes),
        );
        for vector in self.vectors.iter().chain(&self.sparse_vectors) {
            let name = if vector.name.is_empty() {
                "(default)"
            } else {
                &vector.name
            };
            println!(
                "vector {name}: storage {} disk / {} ram, index {} disk / {} ram",
                mib(vector.storage.disk_bytes),
                mib(vector.storage.ram_bytes),
                mib(vector.index.disk_bytes),
                mib(vector.index.ram_bytes),
            );
        }
        println!(
            "payload: {} disk, {} ram",
            mib(self.payload.disk_bytes),
            mib(self.payload.ram_bytes)
        );
        for field in &self.payload_index {
            println!(
                "payload index {}: {} disk, {} ram",
                field.name,
                mib(field.usage.disk_bytes),
                mib(field.usage.ram_bytes)
            );
        }
        if let Some(process) = self.process {
            println!(
                "process: {} resident, {} allocated (server-wide)",
                mib(process.resident_bytes),
                mib(process.allocated_bytes)
            );
        }
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

// ------------------------- wire types -------------------------------------

#[derive(Debug, Deserialize)]
struct RestEnvelope<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct CollectionMemory {
    total: Usage,
    #[serde(default)]
    vectors: Vec<VectorUsage>,
    #[serde(default)]
    sparse_vectors: Vec<VectorUsage>,
    #[serde(default)]
    payload: Usage,
    #[serde(default)]
    payload_index: Vec<NamedUsage>,
    #[serde(default)]
    other: BTreeMap<String, Usage>,
}

#[derive(Debug, Deserialize)]
struct Telemetry {
    memory: Option<ProcessMemory>,
}

// ------------------------- fetching ---------------------------------------

/// Sample memory and disk usage for `collection`.
///
/// The process-wide `/telemetry` half is best-effort: it is unavailable when the
/// server runs without jemalloc, and absent then rather than failing the sample.
pub fn fetch(rest_url: &str, collection: &str, api_key: Option<&str>) -> Result<MemoryReport> {
    let base = rest_url.trim_end_matches('/');
    let memory: CollectionMemory =
        get(&format!("{base}/collections/{collection}/memory"), api_key)?;
    let process = get::<Telemetry>(&format!("{base}/telemetry?details_level=1"), api_key)
        .ok()
        .and_then(|telemetry| telemetry.memory);

    Ok(MemoryReport {
        total: memory.total,
        vectors: memory.vectors,
        sparse_vectors: memory.sparse_vectors,
        payload: memory.payload,
        payload_index: memory.payload_index,
        other: memory.other,
        process,
    })
}

fn get<T: serde::de::DeserializeOwned>(url: &str, api_key: Option<&str>) -> Result<T> {
    let agent = ureq::Agent::new_with_defaults();
    let mut request = agent.get(url);
    if let Some(key) = api_key {
        request = request.header("api-key", key);
    }

    let body = request
        .call()
        .with_context(|| format!("failed to GET {url}"))?
        .into_body()
        .read_to_string()
        .with_context(|| format!("failed to read response from {url}"))?;

    let envelope: RestEnvelope<T> =
        serde_json::from_str(&body).with_context(|| format!("failed to parse {url}"))?;
    Ok(envelope.result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_server_response() {
        let text = include_str!("../tests/fixtures/collection_memory.json");
        let envelope: RestEnvelope<CollectionMemory> = serde_json::from_str(text).unwrap();
        let memory = envelope.result;

        assert_eq!(memory.total.disk_bytes, 161554912);
        assert_eq!(memory.vectors.len(), 1);
        assert_eq!(memory.vectors[0].name, "");
        assert_eq!(memory.vectors[0].index.disk_bytes, 7124194);
        assert_eq!(memory.payload_index[0].name, "color");
        assert_eq!(memory.payload_index[0].usage.ram_bytes, 2817048);
        assert!(memory.other.contains_key("id_tracker"));
    }
}
