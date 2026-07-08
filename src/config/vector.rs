//! Dense and sparse vector declarations and their value sources.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::collection::QuantizationConfig;
use super::string_or_struct;
use crate::dataset::DatasetConfig;

// ----------------------------- Dense vectors -----------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorConfig {
    /// Vector name. Omit for the unnamed default vector.
    #[serde(default)]
    pub name: Option<String>,
    pub size: u64,
    #[serde(default)]
    pub distance: DistanceKind,
    #[serde(default)]
    pub datatype: DatatypeKind,
    pub on_disk: Option<bool>,
    pub multivector: Option<MultivectorConfig>,
    pub quantization: Option<QuantizationConfig>,
    #[serde(default, deserialize_with = "string_or_struct")]
    pub source: VectorSource,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceKind {
    #[default]
    Cosine,
    Dot,
    Euclid,
    Manhattan,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DatatypeKind {
    #[default]
    Float32,
    Float16,
    Uint8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultivectorConfig {
    #[serde(default)]
    pub comparator: ComparatorKind,
    /// Number of sub-vectors to generate per point.
    pub count: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparatorKind {
    #[default]
    MaxSim,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum VectorSource {
    #[default]
    Random,
    File {
        path: String,
        #[serde(default)]
        strategy: FileStrategy,
    },
    /// vector-db-benchmark dataset (specified inline in the source definition).
    Dataset {
        #[serde(flatten)]
        dataset: DatasetConfig,
    },
}

impl FromStr for VectorSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(VectorSource::Random),
            other => Err(format!(
                "unknown vector source {other:?}; expected \"random\" or a map with `type: file|dataset`"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileStrategy {
    #[default]
    RandomSample,
    FromStart,
}

// ----------------------------- Sparse vectors ----------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseVectorConfig {
    pub name: String,
    #[serde(default)]
    pub datatype: DatatypeKind,
    #[serde(default)]
    pub on_disk: bool,
    #[serde(default, deserialize_with = "string_or_struct")]
    pub source: SparseSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseSource {
    #[serde(default, rename = "type")]
    pub kind: SparseKind,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_sparse_length")]
    pub length: usize,
    #[serde(default)]
    pub distribution: DistributionKind,
    /// vector-db-benchmark dataset (specified inline under `dataset`).
    #[serde(default)]
    pub dataset: Option<DatasetConfig>,
}

impl Default for SparseSource {
    fn default() -> Self {
        SparseSource {
            kind: SparseKind::Random,
            vocab_size: default_vocab_size(),
            length: default_sparse_length(),
            distribution: DistributionKind::default(),
            dataset: None,
        }
    }
}

impl FromStr for SparseSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(SparseSource::default()),
            other => Err(format!(
                "unknown sparse source {other:?}; expected \"random\""
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum SparseKind {
    #[default]
    Random,
    Dataset,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum DistributionKind {
    #[default]
    Uniform,
    Zipf,
}

fn default_vocab_size() -> usize {
    1000
}
fn default_sparse_length() -> usize {
    100
}
