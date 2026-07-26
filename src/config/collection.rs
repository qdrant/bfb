//! Collection-level settings: params, HNSW, optimizers, quantization, sharding.

use serde::{Deserialize, Serialize};

use super::payload::{PayloadConfig, PayloadSource};
use super::vector::{SparseVectorConfig, VectorConfig};
use super::{default_collection_name, default_custom, default_one, default_true};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionConfig {
    #[serde(default = "default_collection_name")]
    pub name: String,
    #[serde(default)]
    pub id: IdType,
    #[serde(default = "default_true")]
    pub on_disk_payload: bool,
    pub shard_number: Option<u32>,
    #[serde(default = "default_one")]
    pub replication_factor: u32,
    #[serde(default = "default_one")]
    pub write_consistency_factor: u32,
    pub sharding: Option<ShardingConfig>,
    pub hnsw: Option<HnswConfig>,
    pub optimizers: Option<OptimizersConfig>,
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub vectors: Vec<VectorConfig>,
    #[serde(default)]
    pub sparse_vectors: Vec<SparseVectorConfig>,
    /// Payload-wide settings, notably the whole-payload `source`.
    #[serde(default)]
    pub payload: PayloadSection,
    /// Payload field declarations: value generation and/or which fields to index.
    #[serde(default)]
    pub fields: Vec<PayloadConfig>,
}

/// Payload-wide settings (`collection.payload`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadSection {
    /// Whole-payload source: when set (to `type: dataset`), every point's entire
    /// payload object is loaded from the dataset's `payloads.jsonl`. `fields`
    /// entries then only need to declare which fields to index (they may omit
    /// their own `source`); fields not listed are uploaded but left unindexed.
    #[serde(default)]
    pub source: Option<PayloadSource>,
    /// Memory placement of the payload storage. Supersedes `on_disk_payload`.
    #[serde(default)]
    pub memory: Option<MemoryKind>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IdType {
    #[default]
    Integer,
    Uuid,
}

/// Memory placement of a component's data (Qdrant 1.19+). Data is always
/// persisted on disk; this only controls how it is held in RAM. Supersedes the
/// older `on_disk` / `always_ram` booleans, which stay available for older
/// servers — when both are given, `memory` wins.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Not pre-loaded from disk; cached with usage.
    Cold,
    /// Pre-loaded into disk-cache RAM on start, may be evicted under pressure.
    Cached,
    /// Loaded in RAM and never evicted. Unsupported for dense vector storage
    /// and payload storage.
    Pinned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardingConfig {
    #[serde(default = "default_custom")]
    pub method: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswConfig {
    pub m: Option<u64>,
    pub payload_m: Option<u64>,
    pub ef_construct: Option<u64>,
    pub full_scan_threshold: Option<u64>,
    #[serde(default)]
    pub on_disk: bool,
    #[serde(default)]
    pub inline_storage: bool,
    /// Memory placement of the HNSW graph. Supersedes `on_disk`.
    #[serde(default)]
    pub memory: Option<MemoryKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizersConfig {
    pub default_segment_number: Option<u64>,
    pub indexing_threshold: Option<u64>,
    pub memmap_threshold: Option<u64>,
    pub max_segment_size: Option<u64>,
    #[serde(default)]
    pub prevent_unoptimized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    #[serde(rename = "type")]
    pub kind: QuantKind,
    #[serde(default)]
    pub always_ram: bool,
    /// Memory placement of the quantized vectors. Supersedes `always_ram`.
    #[serde(default)]
    pub memory: Option<MemoryKind>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum QuantKind {
    None,
    Scalar,
    Binary,
    #[serde(rename = "binary-2bit")]
    Binary2bit,
    #[serde(rename = "binary-1.5bit")]
    Binary15bit,
    #[serde(rename = "turbo-1bit")]
    Turbo1bit,
    #[serde(rename = "turbo-1.5bit")]
    Turbo15bit,
    #[serde(rename = "turbo-2bit")]
    Turbo2bit,
    #[serde(rename = "turbo-4bit")]
    Turbo4bit,
    #[serde(rename = "product-x4")]
    ProductX4,
    #[serde(rename = "product-x8")]
    ProductX8,
    #[serde(rename = "product-x16")]
    ProductX16,
    #[serde(rename = "product-x32")]
    ProductX32,
    #[serde(rename = "product-x64")]
    ProductX64,
}
