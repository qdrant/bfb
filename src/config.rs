//! YAML collection-shape configuration for `bfb upload --file config.yaml`.
//!
//! The config file describes only the *shape* of the data (collection params +
//! how each field's values are generated). The *how* of uploading (number of
//! points, batch size, threads, parallelism, uri, …) stays on the CLI.
//!
//! See `BENCHMARK_ROADMAP.md` §2 for the schema rationale and `examples/`.

use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// Top-level document: `{ collection: { ... } }`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadConfig {
    pub collection: CollectionConfig,
}

#[derive(Debug, Clone, Deserialize)]
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
    #[serde(default)]
    pub payloads: Vec<PayloadConfig>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IdType {
    #[default]
    Integer,
    Uuid,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardingConfig {
    #[serde(default = "default_custom")]
    pub method: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizersConfig {
    pub default_segment_number: Option<u64>,
    pub indexing_threshold: Option<u64>,
    pub memmap_threshold: Option<u64>,
    pub max_segment_size: Option<u64>,
    #[serde(default)]
    pub prevent_unoptimized: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    #[serde(rename = "type")]
    pub kind: QuantKind,
    #[serde(default)]
    pub always_ram: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
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

// ----------------------------- Dense vectors -----------------------------

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceKind {
    #[default]
    Cosine,
    Dot,
    Euclid,
    Manhattan,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DatatypeKind {
    #[default]
    Float32,
    Float16,
    Uint8,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultivectorConfig {
    #[serde(default)]
    pub comparator: ComparatorKind,
    /// Number of sub-vectors to generate per point.
    pub count: usize,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparatorKind {
    #[default]
    MaxSim,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum VectorSource {
    #[default]
    Random,
    File {
        path: String,
        #[serde(default)]
        strategy: FileStrategy,
    },
}

impl FromStr for VectorSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(VectorSource::Random),
            other => Err(format!(
                "unknown vector source {other:?}; expected \"random\" or a map with `type: file`"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileStrategy {
    #[default]
    RandomSample,
    FromStart,
}

// ----------------------------- Sparse vectors ----------------------------

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseSource {
    /// Accepted for schema symmetry / `type: random`; only `random` is supported.
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    pub kind: SparseKind,
    #[serde(default = "default_sparse_dim")]
    pub dim: usize,
    #[serde(default = "default_sparsity")]
    pub sparsity: f64,
    #[serde(default)]
    pub distribution: DistributionKind,
}

impl Default for SparseSource {
    fn default() -> Self {
        SparseSource {
            kind: SparseKind::Random,
            dim: default_sparse_dim(),
            sparsity: default_sparsity(),
            distribution: DistributionKind::default(),
        }
    }
}

impl FromStr for SparseSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(SparseSource::default()),
            other => Err(format!("unknown sparse source {other:?}; expected \"random\"")),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum SparseKind {
    #[default]
    Random,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum DistributionKind {
    #[default]
    Uniform,
    Zipf,
}

// -------------------------------- Payloads -------------------------------

#[derive(Debug, Clone, Deserialize)]
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
    #[serde(default)]
    pub is_tenant: bool,
    #[serde(default)]
    pub is_principal: bool,
    /// Integer payloads: also build a range index.
    #[serde(default = "default_true")]
    pub range_index: bool,
    /// Text payloads: tokenizer.
    pub tokenizer: Option<TokenizerKind>,
    #[serde(default, deserialize_with = "string_or_struct")]
    pub source: PayloadSource,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    Word,
    Whitespace,
    Prefix,
    Multilingual,
}

/// All payload value-generation parameters. Which keys apply depends on the
/// payload `type`; irrelevant keys are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadSource {
    #[serde(default, rename = "type")]
    pub kind: PayloadSourceKind,
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

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PayloadSourceKind {
    #[default]
    Random,
    RandomClusters,
    Now,
}

// ------------------------------ Loading / validation ---------------------

/// Read and validate a YAML config file.
pub fn load(path: &str) -> Result<UploadConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {path}"))?;
    let config: UploadConfig =
        serde_yaml::from_str(&text).with_context(|| format!("failed to parse config file {path}"))?;
    config.validate()?;
    Ok(config)
}

impl UploadConfig {
    pub fn validate(&self) -> Result<()> {
        let c = &self.collection;

        if c.vectors.is_empty() && c.sparse_vectors.is_empty() {
            bail!("config must define at least one dense or sparse vector");
        }

        // Dense vector names: at most one unnamed; the rest must be named & unique.
        let unnamed = c.vectors.iter().filter(|v| v.name.is_none()).count();
        if unnamed > 1 {
            bail!("at most one dense vector may omit `name` (the default vector)");
        }
        if unnamed == 1 && c.vectors.len() > 1 {
            // The unnamed default vector cannot coexist with named ones in a map.
            bail!(
                "when multiple dense vectors are defined, every vector must have a `name`"
            );
        }

        let mut names = std::collections::HashSet::new();
        for v in &c.vectors {
            if let Some(n) = &v.name
                && !names.insert(n.clone())
            {
                bail!("duplicate vector name {n:?}");
            }
            if let VectorSource::File { path, .. } = &v.source
                && !path.starts_with("s3://")
                && !path.starts_with("http")
                && !Path::new(path).exists()
            {
                bail!("vector source file not found: {path}");
            }
            if let Some(mv) = &v.multivector
                && mv.count == 0
            {
                bail!("multivector.count must be > 0");
            }
        }
        for s in &c.sparse_vectors {
            if !names.insert(s.name.clone()) {
                bail!("sparse vector name {:?} collides with another vector", s.name);
            }
            if !(0.0..=1.0).contains(&s.source.sparsity) {
                bail!("sparse sparsity must be in [0, 1], got {}", s.source.sparsity);
            }
        }

        let mut payload_names = std::collections::HashSet::new();
        for p in &c.payloads {
            if !payload_names.insert(p.name.clone()) {
                bail!("duplicate payload name {:?}", p.name);
            }
            if let Some(c) = p.source.clusters
                && c == 0
            {
                bail!("payload {:?}: clusters must be > 0", p.name);
            }
        }

        if let Some(sh) = &c.sharding
            && sh.method != "custom"
        {
            bail!("only `custom` sharding method is supported, got {:?}", sh.method);
        }

        Ok(())
    }
}

// --------------------------- serde helpers -------------------------------

/// Deserialize a value that may be either a bare string (shorthand, parsed via
/// `FromStr`) or a full map (parsed via `Deserialize`). Standard serde idiom.
fn string_or_struct<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = String>,
    D: Deserializer<'de>,
{
    struct StringOrStruct<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrStruct<T>
    where
        T: Deserialize<'de> + FromStr<Err = String>,
    {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a map")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<T, E> {
            FromStr::from_str(value).map_err(de::Error::custom)
        }

        fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
            Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))
        }
    }

    deserializer.deserialize_any(StringOrStruct(PhantomData))
}

fn default_true() -> bool {
    true
}
fn default_one() -> u32 {
    1
}
fn default_collection_name() -> String {
    "benchmark".to_string()
}
fn default_custom() -> String {
    "custom".to_string()
}
fn default_sparse_dim() -> usize {
    1000
}
fn default_sparsity() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let yaml = r#"
collection:
  name: test
  vectors:
    - size: 128
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.collection.name, "test");
        assert_eq!(cfg.collection.vectors.len(), 1);
        assert_eq!(cfg.collection.vectors[0].size, 128);
        assert!(matches!(cfg.collection.vectors[0].source, VectorSource::Random));
        assert!(matches!(cfg.collection.id, IdType::Integer));
    }

    #[test]
    fn source_shorthand_and_map() {
        let yaml = r#"
collection:
  vectors:
    - name: a
      size: 64
      source: random
    - name: b
      size: 64
      source:
        type: file
        path: /etc/hostname
        strategy: from-start
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(cfg.collection.vectors[0].source, VectorSource::Random));
        match &cfg.collection.vectors[1].source {
            VectorSource::File { path, strategy } => {
                assert_eq!(path, "/etc/hostname");
                assert!(matches!(strategy, FileStrategy::FromStart));
            }
            _ => panic!("expected file source"),
        }
    }

    #[test]
    fn parses_payloads_and_quantization() {
        let yaml = r#"
collection:
  quantization:
    type: turbo-4bit
    always_ram: true
  vectors:
    - size: 256
  payloads:
    - name: color
      type: keyword
      source:
        type: random
        cardinality: 100
        values_per_point: 3
    - name: blob
      type: text
      index: false
      source: random
    - name: loc
      type: geo
      source:
        type: random-clusters
        clusters: 10
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.collection.quantization.as_ref().unwrap().kind, QuantKind::Turbo4bit);
        assert_eq!(cfg.collection.payloads.len(), 3);
        assert_eq!(cfg.collection.payloads[0].source.cardinality, Some(100));
        assert!(!cfg.collection.payloads[1].index);
        assert_eq!(cfg.collection.payloads[2].source.kind, PayloadSourceKind::RandomClusters);
    }

    #[test]
    fn rejects_unknown_field() {
        let yaml = r#"
collection:
  vectors:
    - size: 128
      bogus: 1
"#;
        assert!(serde_yaml::from_str::<UploadConfig>(yaml).is_err());
    }

    #[test]
    fn rejects_multiple_unnamed_vectors() {
        let yaml = r#"
collection:
  vectors:
    - size: 64
    - size: 64
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_empty_collection() {
        let yaml = "collection: {}\n";
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }
}
