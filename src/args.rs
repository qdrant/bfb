use std::{fmt, str};

use clap::{ArgAction, Parser};
use qdrant_client::qdrant;

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum QuantizationArg {
    #[default]
    None,
    Binary,
    Binary2bit,
    Binary1p5bit,
    Turbo1bit,
    Turbo1p5bit,
    Turbo2bit,
    Turbo4bit,
    Scalar,
    ProductX4,
    ProductX8,
    ProductX16,
    ProductX32,
    ProductX64,
}

/// Big F*cking Benchmark tool for stress-testing Qdrant
#[derive(Parser, Debug, Clone)]
#[clap(version, about)]
#[clap(after_help = "\
   Integers can be suffixed with k/M/G/T/ki/Mi/Gi/Ti, \
   and underscores can be inserted to make them more readable, \
   e.g., `--num-vectors 100k --offset 1_000_000`.\
")]
pub struct Args {
    /// Qdrant service URI
    #[clap(long, default_value = "http://localhost:6334")]
    pub uri: Vec<String>,

    /// Source of data to upload - fbin file. Random if not specified
    #[clap(long)]
    pub fbin: Option<String>,

    /// Number of points to upload
    #[clap(short, long, default_value = "100_000", value_parser = parse_number)]
    pub num_vectors: usize,

    /// Number of named dense vectors per point
    #[clap(long, default_value_t = 1, value_parser = parse_number)]
    pub vectors_per_point: usize,

    #[clap(short, long, default_value_t = 0, value_parser = parse_number)]
    pub offset: usize,

    /// If set, will randomly upsert/override vector ids within range [offset, max_id)
    #[clap(short, long, value_parser = parse_number)]
    pub max_id: Option<usize>,

    /// Number of dimensions in each dense vector or max dimension for sparse vectors
    #[clap(short, long, default_value_t = 128, value_parser = parse_number)]
    pub dim: usize,

    /// Number of worker threads to use
    #[clap(short, long, default_value_t = 2, value_parser = parse_number)]
    pub threads: usize,

    /// Number of parallel requests to send (ignored when --rps is set)
    #[clap(short, long, default_value_t = 2, value_parser = parse_number)]
    pub parallel: usize,

    /// Target requests per second. When set, requests are sent at a fixed rate
    /// regardless of how many are currently in-flight (simulates real user traffic).
    /// This overrides --parallel for concurrency control.
    #[clap(long, value_name = "RATE")]
    pub rps: Option<f64>,

    /// Number of connections to open from the client to the server
    #[clap(short, long, default_value_t = 1, value_parser = parse_number)]
    pub connections: usize,

    /// Batch size for updates, in number of points. [default=100]
    #[clap(short, long, value_name = "POINTS", default_value_t = 100, value_parser = parse_number)]
    pub batch_size: usize,

    /// Batch size for searches, in number of queries per batch.
    #[clap(long, value_name = "QUERIES", default_value_t = 1, value_parser = parse_number)]
    pub search_batch_size: usize,

    /// Throttle updates and searches, in batches/searches per second. [default=no throttling]
    #[clap(long, short = 'T', value_name = "RPS")]
    pub throttle: Option<f32>,

    /// Skip creating a collection
    #[clap(long, default_value_t = false)]
    pub skip_create: bool,

    /// Create if not exists. Avoid re-creating collection
    #[clap(long, default_value_t = false)]
    pub create_if_missing: bool,

    /// Skip wait until collection is indexed after upload
    #[clap(long, default_value_t = false)]
    pub skip_wait_index: bool,

    /// Skip uploading new points
    #[clap(long, default_value_t = false)]
    pub skip_upload: bool,

    /// Skip setting up collections.
    /// Implies --skip-create --skip-upload --skip-wait-index
    #[clap(long, default_value_t = false)]
    pub skip_setup: bool,

    /// Perform search
    #[clap(long, default_value_t = false)]
    pub search: bool,

    /// Perform search without approximation
    #[clap(long, default_value_t = false)]
    pub search_exact: bool,

    /// Prefetch search
    #[clap(long, value_parser = parse_number)]
    pub prefetch: Option<usize>,

    /// Perform scroll
    #[clap(long, default_value_t = false)]
    pub scroll: bool,

    /// Search limit
    #[clap(long, default_value_t = 10, value_parser = parse_number)]
    pub search_limit: usize,

    /// Store results to csv
    #[clap(long)]
    pub json: Option<String>,

    /// Number of 9 digits to show in p99* results
    #[clap(long, long, default_value_t = 2, value_parser = parse_number)]
    pub p9: usize,

    /// Name of the collection to use
    #[clap(long, default_value = "benchmark")]
    pub collection_name: String,

    /// Distance function used for comparing vectors
    #[clap(long, default_value = "Cosine")]
    pub distance: String,

    /// Vector datatypes (Uint8, Float16, Float32)
    #[clap(long)]
    pub datatype: Option<String>,

    /// Store vectors on disk
    #[clap(long, value_parser = parse_number)]
    pub mmap_threshold: Option<usize>,

    /// Index vectors on disk
    #[clap(long, value_parser = parse_number)]
    pub indexing_threshold: Option<usize>,

    /// Number of segments
    #[clap(long, value_parser = parse_number)]
    pub segments: Option<usize>,

    /// Do not create segments larger this size (in kilobytes).
    #[clap(long, value_parser = parse_number)]
    pub max_segment_size: Option<usize>,

    /// On disk payload
    #[clap(long, default_value_t = true, action = ArgAction::Set)]
    pub on_disk_payload: bool,

    /// On disk payload
    #[clap(long, default_value_t = false)]
    pub on_disk_payload_index: bool,

    /// On disk index
    #[clap(long)]
    pub on_disk_index: Option<bool>,

    /// On disk vectors
    #[clap(long)]
    pub on_disk_vectors: Option<bool>,

    /// Log requests if the take longer than this
    #[clap(long, default_value_t = 0.1)]
    pub timing_threshold: f64,

    /// Use UUIDs instead of sequential ids
    #[clap(long, default_value_t = false)]
    pub uuids: bool,

    /// Skip field indices creation if payloads are not empty
    #[clap(long, default_value_t = false)]
    pub skip_field_indices: bool,

    /// Use keyword payloads. Defines how many different keywords there are in the payload
    #[clap(long, short = 'k', value_parser = parse_number)]
    pub keywords: Vec<usize>,

    /// Multiplies the length of keyword payload values by a given factor. Can be used to test larger keyword payloads.
    ///
    /// Note: This must be set for both, uspertions and searches (in case they're running in parallel) to prevent empty results due to different keywords being used.
    #[clap(long, value_parser = parse_number, default_value_t = 1)]
    pub keywords_length_multiplier: usize,

    /// Maximum number of keywords per point
    #[clap(long, value_parser = parse_number, default_value_t = 1)]
    pub max_keywords: usize,

    /// Use float payloads
    #[clap(long)]
    pub float_payloads: Vec<bool>,

    /// Match any count
    #[clap(long, value_parser = parse_number)]
    pub match_any: Option<usize>,

    /// Use integer payloads
    #[clap(long, value_parser = parse_number)]
    pub int_payloads: Vec<usize>,

    /// Whether to enable the range index for the integer payloads
    #[clap(long, default_value_t = false)]
    pub int_payloads_range: bool,

    /// Maximum number of integer payloads per point
    #[clap(long, value_parser = parse_number, default_value_t = 1)]
    pub max_int_payloads: usize,

    #[clap(long, default_value_t = false)]
    pub uuid_payloads: bool,

    /// Generate true/false payloads
    #[clap(long, default_value_t = false)]
    pub bool_payloads: bool,

    /// Use geo payloads
    #[clap(long, default_value_t = false)]
    pub geo_payloads: bool,

    /// generate text-like payloads
    #[clap(long, default_value_t = false)]
    pub text_payloads: bool,

    /// Length of the text-like payloads
    #[clap(long, value_parser = parse_number)]
    pub text_payload_length: Option<usize>,

    /// Vocabulary size for text-like payloads
    #[clap(long, value_parser = parse_number)]
    pub text_payload_vocabulary: Option<usize>,

    /// Add payload with the current timestamp to all points
    #[clap(long)]
    pub timestamp_payload: bool,

    /// Use separate request to set payload on just upserted points
    #[clap(long, default_value_t = false)]
    pub set_payload: bool,

    /// `hnsw_ef_construct` parameter used during index
    #[clap(long, value_parser = parse_number)]
    pub hnsw_ef_construct: Option<usize>,

    /// `hnsw_m` parameter used during index
    #[clap(long, value_parser = parse_number)]
    pub hnsw_m: Option<usize>,

    /// `hnsw_payload_m` parameter used during index
    #[clap(long, value_parser = parse_number)]
    pub hnsw_payload_m: Option<usize>,

    /// `hnsw_ef` parameter used during search
    #[clap(long, value_parser = parse_number)]
    pub search_hnsw_ef: Option<usize>,

    /// Store HNSW links inline with vectors (hnsw_config.inline_storage)
    #[clap(long, default_value_t = false)]
    pub hnsw_inline_storage: bool,

    /// Whether to request payload in search results
    #[clap(long, default_value_t = false)]
    pub search_with_payload: bool,

    /// Whether to request vectors in search results
    #[clap(long, default_value_t = false)]
    pub search_with_vectors: bool,

    /// Wait on upsert
    #[clap(long, default_value_t = false)]
    pub wait_on_upsert: bool,

    /// Prevent unoptimized segments from being used for search
    #[clap(long, default_value_t = false)]
    pub prevent_unoptimized: bool,

    /// Replication factor
    #[clap(long, default_value_t = 1, value_parser = parse_number)]
    pub replication_factor: usize,

    /// Number of shards in the collection
    #[clap(long, value_parser = parse_number)]
    pub shards: Option<usize>,

    /// Write consistency factor to use for collection creation
    #[clap(long, default_value_t = 1, value_parser = parse_number)]
    pub write_consistency_factor: usize,

    /// Write ordering parameter to use for all write requests
    #[clap(long)]
    pub write_ordering: Option<WriteOrdering>,

    /// Read consistency parameter to use for all read requests
    #[clap(long)]
    pub read_consistency: Option<ReadConsistency>,

    /// Timeout for requests in seconds (applied as both the client channel deadline and the server-side request timeout where supported).
    #[clap(long, value_parser = parse_number)]
    pub timeout: Option<usize>,

    /// Number of retries for each URI on error, 0 for no retries
    #[clap(long = "retry", default_value_t = 0, value_parser = parse_number)]
    pub retries: usize,

    /// Number of seconds between each retry.
    #[clap(long, default_value_t = 0.0, value_name = "SECONDS")]
    pub retry_interval: f32,

    /// Keep going on search error
    #[clap(long, default_value_t = false)]
    pub ignore_errors: bool,

    #[clap(long)]
    pub quantization: Option<QuantizationArg>,

    /// Keep quantized vectors in memory
    #[clap(long)]
    pub quantization_in_ram: Option<bool>,

    /// Enable quantization re-score during search
    #[clap(long)]
    pub quantization_rescore: Option<bool>,

    /// Quantization oversampling factor
    #[clap(long)]
    pub quantization_oversampling: Option<f64>,

    /// Delay between requests in milliseconds
    #[clap(long, value_parser = parse_number)]
    pub delay: Option<usize>,

    /// Skip un-indexed segments during search
    #[clap(long)]
    pub indexed_only: Option<bool>,

    /// Whether to use sparse vectors and with how much sparsity
    #[clap(long, value_name = "SPARSITY")]
    pub sparse_vectors: Option<f64>,

    /// Number of named sparse vectors per point
    #[clap(long, default_value_t = 1)]
    pub sparse_vectors_per_point: usize,

    /// Whether to set dense vectors as multivectors
    #[clap(long)]
    pub multivector_size: Option<usize>,

    /// Max dimension for sparse vectors (overrides --dim)
    #[clap(long, value_parser = parse_number)]
    pub sparse_dim: Option<usize>,

    /// Path to the jsonl file to save update timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long)]
    pub jsonl_updates: Option<String>,

    /// Path to the jsonl file to save search timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long)]
    pub jsonl_searches: Option<String>,

    /// Path to the jsonl file to save rps timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long)]
    pub jsonl_rps: Option<String>,

    /// Use timestamp instead of relative time in jsonl
    /// Default is relative time
    #[clap(long)]
    pub absolute_time: Option<bool>,

    /// Use custom sharding for collection and upsert points to the specified sharding key
    #[clap(long)]
    pub shard_key: Option<String>,

    /// Use tenant optimization for field index.
    #[clap(long)]
    pub tenants: Option<bool>,

    /// Use a custom UUID as filter when searching.
    #[clap(long)]
    pub uuid_query: Option<String>,

    /// Bench for search quality / accurracy too.
    #[clap(long)]
    pub search_quality: bool,

    /// Set a custom full-scan threshold.
    #[clap(long)]
    pub full_scan_threshold: Option<usize>,
}

impl Args {
    pub fn is_uint8_datatype(&self) -> bool {
        self.datatype
            .as_ref()
            .is_some_and(|x| x == qdrant::Datatype::Uint8.as_str_name())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum WriteOrdering {
    Weak,
    Medium,
    Strong,
}

impl fmt::Display for WriteOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self {
            Self::Weak => "Weak",
            Self::Medium => "Medium",
            Self::Strong => "Strong",
        };

        str.fmt(f)
    }
}

impl str::FromStr for WriteOrdering {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "Weak" => Ok(Self::Weak),
            "Medium" => Ok(Self::Medium),
            "Strong" => Ok(Self::Strong),
            _ => Err(anyhow::format_err!(
                "invalid WriteOrdering value {str}, \
                 valid values are Weak, Medium or Strong"
            )),
        }
    }
}

impl From<WriteOrdering> for qdrant::WriteOrdering {
    fn from(ordering: WriteOrdering) -> Self {
        qdrant::WriteOrdering {
            r#type: ordering.into(),
        }
    }
}

impl From<WriteOrdering> for i32 {
    fn from(ordering: WriteOrdering) -> Self {
        qdrant::WriteOrderingType::from(ordering) as _
    }
}

impl From<WriteOrdering> for qdrant::WriteOrderingType {
    fn from(ordering: WriteOrdering) -> Self {
        match ordering {
            WriteOrdering::Weak => Self::Weak,
            WriteOrdering::Medium => Self::Medium,
            WriteOrdering::Strong => Self::Strong,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ReadConsistency {
    Type(ReadConsistencyType),
    Factor(u64),
}

impl fmt::Display for ReadConsistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(consistency) => consistency.fmt(f),
            Self::Factor(factor) => factor.fmt(f),
        }
    }
}

impl str::FromStr for ReadConsistency {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        if let Ok(consistency) = str.parse() {
            return Ok(Self::Type(consistency));
        }

        if let Ok(factor) = str.parse() {
            return Ok(Self::Factor(factor));
        }

        Err(anyhow::format_err!(
            "invalid ReadConsistency value {str}, \
             valid values are All, Majority, Quorum or a positive integer number"
        ))
    }
}

impl From<ReadConsistency> for qdrant::read_consistency::Value {
    fn from(consistency: ReadConsistency) -> Self {
        match consistency {
            ReadConsistency::Type(consistency) => consistency.into(),
            ReadConsistency::Factor(factor) => qdrant::read_consistency::Value::Factor(factor),
        }
    }
}

impl From<ReadConsistency> for qdrant::ReadConsistency {
    fn from(consistency: ReadConsistency) -> Self {
        let consistency = match consistency {
            ReadConsistency::Type(consistency) => consistency.into(),
            ReadConsistency::Factor(factor) => qdrant::read_consistency::Value::Factor(factor),
        };

        qdrant::ReadConsistency {
            value: consistency.into(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ReadConsistencyType {
    All,
    Majority,
    Quorum,
}

impl fmt::Display for ReadConsistencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self {
            Self::All => "All",
            Self::Majority => "Majority",
            Self::Quorum => "Quorum",
        };

        str.fmt(f)
    }
}

impl str::FromStr for ReadConsistencyType {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "All" => Ok(Self::All),
            "Majority" => Ok(Self::Majority),
            "Quorum" => Ok(Self::Quorum),
            _ => Err(anyhow::format_err!(
                "invalid ReadConsistencyType value {str}, \
                 valid values are All, Majority or Quorum"
            )),
        }
    }
}

impl From<ReadConsistencyType> for qdrant::read_consistency::Value {
    fn from(consistency: ReadConsistencyType) -> Self {
        qdrant::read_consistency::Value::Type(consistency.into())
    }
}

impl From<ReadConsistencyType> for i32 {
    fn from(consistency: ReadConsistencyType) -> Self {
        qdrant::ReadConsistencyType::from(consistency) as _
    }
}

impl From<ReadConsistencyType> for qdrant::ReadConsistencyType {
    fn from(consistency: ReadConsistencyType) -> Self {
        match consistency {
            ReadConsistencyType::All => qdrant::ReadConsistencyType::All,
            ReadConsistencyType::Majority => qdrant::ReadConsistencyType::Majority,
            ReadConsistencyType::Quorum => qdrant::ReadConsistencyType::Quorum,
        }
    }
}

fn parse_number(n: &str) -> Result<usize, String> {
    parse_number_impl(n)
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| format!("Invalid number: {n}"))
}

fn parse_number_impl(n: &str) -> Option<u64> {
    let mut chars = n.chars();
    let mut result = chars.next()?.to_digit(10)? as u64;

    while let Some(c) = chars.next() {
        if let Some(v) = c.to_digit(10) {
            result = result.checked_mul(10)?.checked_add(v as u64)?;
        } else if c != '_' {
            let power = "kMGT".find(c)? as u32 + 1;
            let multiplier = match chars.next() {
                Some('i') => 1024u64.pow(power),
                Some(_) => return None,
                None => 1000u64.pow(power),
            };
            return result.checked_mul(multiplier);
        }
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_ordering_from_str() {
        assert!(matches!(
            "Weak".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Weak
        ));
        assert!(matches!(
            "Medium".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Medium
        ));
        assert!(matches!(
            "Strong".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Strong
        ));
        assert!("weak".parse::<WriteOrdering>().is_err());
        assert!("".parse::<WriteOrdering>().is_err());
    }

    #[test]
    fn test_write_ordering_display_roundtrip() {
        for ordering in [
            WriteOrdering::Weak,
            WriteOrdering::Medium,
            WriteOrdering::Strong,
        ] {
            let s = ordering.to_string();
            let parsed: WriteOrdering = s.parse().unwrap();
            assert_eq!(ordering.to_string(), parsed.to_string());
        }
    }

    #[test]
    fn test_read_consistency_type_from_str() {
        assert!(matches!(
            "All".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::All
        ));
        assert!(matches!(
            "Majority".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::Majority
        ));
        assert!(matches!(
            "Quorum".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::Quorum
        ));
        assert!("all".parse::<ReadConsistencyType>().is_err());
        assert!("".parse::<ReadConsistencyType>().is_err());
    }

    #[test]
    fn test_read_consistency_type_display_roundtrip() {
        for consistency in [
            ReadConsistencyType::All,
            ReadConsistencyType::Majority,
            ReadConsistencyType::Quorum,
        ] {
            let s = consistency.to_string();
            let parsed: ReadConsistencyType = s.parse().unwrap();
            assert_eq!(consistency.to_string(), parsed.to_string());
        }
    }

    #[test]
    fn test_read_consistency_from_str_with_factor() {
        let parsed: ReadConsistency = "3".parse().unwrap();
        assert!(matches!(parsed, ReadConsistency::Factor(3)));
    }

    #[test]
    fn test_read_consistency_from_str_with_type() {
        let parsed: ReadConsistency = "All".parse().unwrap();
        assert!(matches!(
            parsed,
            ReadConsistency::Type(ReadConsistencyType::All)
        ));
    }

    #[test]
    fn test_parse_number() {
        // Basic numbers
        assert_eq!(parse_number_impl("0"), Some(0));
        assert_eq!(parse_number_impl("42"), Some(42));
        assert_eq!(parse_number_impl("123_456"), Some(123_456));

        // Decimal prefixes
        assert_eq!(parse_number_impl("42k"), Some(42_000));
        assert_eq!(parse_number_impl("42M"), Some(42_000_000));
        assert_eq!(parse_number_impl("42G"), Some(42_000_000_000));
        assert_eq!(parse_number_impl("42T"), Some(42_000_000_000_000));

        // Binary prefixes
        assert_eq!(parse_number_impl("42ki"), Some(42 << 10));
        assert_eq!(parse_number_impl("42Mi"), Some(42 << 20));
        assert_eq!(parse_number_impl("42Gi"), Some(42 << 30));
        assert_eq!(parse_number_impl("42Ti"), Some(42 << 40));

        // Invalid inputs
        assert_eq!(parse_number_impl(""), None);
        assert_eq!(parse_number_impl("abc"), None);
        assert_eq!(parse_number_impl("12a"), None);
        assert_eq!(parse_number_impl("12.3"), None);
        assert_eq!(parse_number_impl("k"), None);
        assert_eq!(parse_number_impl("_k"), None);
        assert_eq!(parse_number_impl("123k_"), None);
        assert_eq!(parse_number_impl("123kk"), None);
        assert_eq!(parse_number_impl("123k_i"), None);
        assert_eq!(parse_number_impl("123ka"), None);

        // Overflows
        assert_eq!(parse_number_impl("18446744073709551615"), Some(u64::MAX));
        assert_eq!(parse_number_impl("18446744073709551616"), None);
        assert_eq!(parse_number_impl("18446744073709551620"), None);
        assert_eq!(parse_number_impl("16777215Ti"), Some(16777215 << 40));
        assert_eq!(parse_number_impl("16777216Ti"), None);
    }
}
