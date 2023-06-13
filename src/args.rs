use std::{fmt, str};

use clap::Parser;
use qdrant_client::qdrant;

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum QuantizationArg {
    #[default]
    None,
    Scalar,
    X4,
    X8,
    X16,
    X32,
    X64,
}

/// Big F*cking Benchmark tool for stress-testing Qdrant
#[derive(Parser, Debug, Clone)]
#[clap(version, about)]
pub struct Args {
    /// Qdrant service URI
    #[clap(long, default_value = "http://localhost:6334")]
    pub uri: Vec<String>,

    /// Source of data to upload - fbin file. Random if not specified
    #[clap(long)]
    pub fbin: Option<String>,

    #[clap(short, long, default_value_t = 100_000)]
    pub num_vectors: usize,

    #[clap(long, default_value_t = 1)]
    pub vectors_per_point: usize,

    /// If set, will use vector ids within range [0, max_id)
    /// To simulate overwriting existing vectors
    #[clap(short, long)]
    pub max_id: Option<usize>,

    #[clap(short, long, default_value_t = 128)]
    pub dim: usize,

    #[clap(short, long, default_value_t = 2)]
    pub threads: usize,

    /// Number of parallel requests to send
    #[clap(short, long, default_value_t = 2)]
    pub parallel: usize,

    #[clap(short, long, default_value_t = 100)]
    pub batch_size: usize,

    /// Skip creation of the collection
    #[clap(long, default_value_t = false)]
    pub skip_create: bool,

    /// If set, after upload will wait until collection is indexed
    #[clap(long, default_value_t = false)]
    pub skip_wait_index: bool,

    /// Perform data upload
    #[clap(long, default_value_t = false)]
    pub skip_upload: bool,

    /// Perform search
    #[clap(long, default_value_t = false)]
    pub search: bool,

    /// Search limit
    #[clap(long, default_value_t = 10)]
    pub search_limit: usize,

    #[clap(long, default_value = "benchmark")]
    pub collection_name: String,

    #[clap(long, default_value = "Cosine")]
    pub distance: String,

    /// Store vectors on disk
    #[clap(long)]
    pub mmap_threshold: Option<usize>,

    /// Index vectors on disk
    #[clap(long)]
    pub indexing_threshold: Option<usize>,

    /// Number of segments
    #[clap(long)]
    pub segments: Option<usize>,

    #[clap(long)]
    pub max_segment_size: Option<usize>,

    /// On disk payload
    #[clap(long, default_value_t = false)]
    pub on_disk_payload: bool,

    /// On disk hnsw
    #[clap(long, default_value_t = false)]
    pub on_disk_hnsw: bool,

    /// Log requests if the take longer than this
    #[clap(long, default_value_t = 0.1)]
    pub timing_threshold: f64,

    /// Use UUIDs instead of sequential ids
    #[clap(long, default_value_t = false)]
    pub uuids: bool,

    /// Use keyword payloads. Defines how many different keywords there are in the payload
    #[clap(long)]
    pub keywords: Option<usize>,

    /// Use separate request to set payload on just upserted points
    #[clap(long, default_value_t = false)]
    pub set_payload: bool,

    /// `hnsw_ef` parameter used during search
    #[clap(long)]
    pub search_hnsw_ef: Option<usize>,

    /// Whether to request payload in search results
    #[clap(long, default_value_t = false)]
    pub search_with_payload: bool,

    /// wait on upsert
    #[clap(long, default_value_t = false)]
    pub wait_on_upsert: bool,

    /// replication factor
    #[clap(long, default_value_t = 1)]
    pub replication_factor: usize,

    #[clap(long)]
    pub shards: Option<usize>,

    /// Write consistency factor to use for collection creation
    #[clap(long, default_value_t = 1)]
    pub write_consistency_factor: usize,

    /// Write ordering parameter to use for all write requests
    #[clap(long)]
    pub write_ordering: Option<WriteOrdering>,

    /// Read consistency parameter to use for all read requests
    #[clap(long)]
    pub read_consistency: Option<ReadConsistency>,

    /// timeout for requests in seconds
    #[clap(long)]
    pub timeout: Option<usize>,

    #[clap(long)]
    pub quantization: Option<QuantizationArg>,

    /// Keep quantized vectors in memory
    #[clap(long)]
    pub quantization_in_ram: Option<bool>,

    /// Enable quantization re-score during search
    #[clap(long)]
    pub quantization_rescore: Option<bool>,
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
