//! Payload field declarations and their value sources.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::collection::MemoryKind;
use super::vector::DistributionKind;
use super::{default_true, option_string_or_struct};
use crate::dataset::DatasetConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: PayloadType,
    /// Create a field index for this payload? `false` ⇒ unindexed filler.
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub on_disk: bool,
    /// Memory placement of the field index. Supersedes `on_disk`.
    #[serde(default)]
    pub memory: Option<MemoryKind>,
    #[serde(default)]
    pub is_tenant: bool,
    #[serde(default)]
    pub is_principal: bool,
    /// Integer payloads: also build a range index.
    #[serde(default = "default_true")]
    pub range_index: bool,
    /// Keyword payloads: build the index with prefix matching enabled, so
    /// searches may use `match_prefix` filters on this field.
    #[serde(default)]
    pub prefix: bool,
    /// Text payloads: tokenizer.
    pub tokenizer: Option<TokenizerKind>,
    /// Per-field value source. May be omitted when `collection.payload.source`
    /// provides the payload object — the entry is then index-only.
    #[serde(default, deserialize_with = "option_string_or_struct")]
    pub source: Option<PayloadSource>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PayloadType {
    Keyword,
    Integer,
    Float,
    Bool,
    Uuid,
    Geo,
    Text,
    Datetime,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    Word,
    Whitespace,
    Prefix,
    Multilingual,
}

/// All payload value-generation parameters. Which keys apply depends on the
/// payload `type`; irrelevant keys are ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadSource {
    #[serde(default, rename = "type")]
    pub kind: PayloadSourceKind,
    /// vector-db-benchmark dataset for payload values (`type: dataset`).
    #[serde(default)]
    pub dataset: Option<DatasetConfig>,
    /// Payload field name inside the dataset schema / `payloads.jsonl`.
    #[serde(default)]
    pub field: Option<String>,
    // keyword
    pub cardinality: Option<usize>,
    pub length_multiplier: Option<usize>,
    // keyword / integer: multivalue
    pub values_per_point: Option<usize>,
    // integer / float / datetime range
    pub min: Option<f64>,
    pub max: Option<f64>,
    // bool
    pub true_ratio: Option<f64>,
    // geo
    pub clusters: Option<usize>,
    // text
    pub vocab_size: Option<usize>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    #[serde(default)]
    pub distribution: DistributionKind,
}

impl FromStr for PayloadSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let kind = match s {
            "random" => PayloadSourceKind::Random,
            "random-clusters" => PayloadSourceKind::RandomClusters,
            "now" => PayloadSourceKind::Now,
            other => return Err(format!("unknown payload source {other:?}")),
        };
        Ok(PayloadSource {
            kind,
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PayloadSourceKind {
    #[default]
    Random,
    RandomClusters,
    Now,
    Dataset,
}
