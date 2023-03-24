use clap::Parser;


#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum QuantizationArg {
    #[default]
    None,
    Scalar
}


/// Big F*cking Benchmark tool for stress-testing Qdrant
#[derive(Parser, Debug, Clone)]
#[clap(version, about)]
pub struct Args {
    /// Qdrant service URI
    #[clap(long, default_value = "http://localhost:6334")]
    pub uri: String,

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

    /// timeout for requests in seconds
    #[clap(long)]
    pub timeout: Option<usize>,

    #[clap(long)]
    pub quantization: Option<QuantizationArg>,

    /// Enable quantization re-score during search
    #[clap(long, default_value_t = false)]
    pub quantization_rescore: bool,
}
