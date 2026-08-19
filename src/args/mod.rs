//! Command-line interface: the [`Args`] struct and its value types.

mod consistency;

pub use consistency::{ReadConsistency, WriteOrdering};

use clap::{ArgAction, Parser, Subcommand};
use qdrant_client::qdrant;

use crate::config::{MemoryKind, QuantKind};

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

impl From<QuantizationArg> for QuantKind {
    fn from(arg: QuantizationArg) -> Self {
        match arg {
            QuantizationArg::None => QuantKind::None,
            QuantizationArg::Binary => QuantKind::Binary,
            QuantizationArg::Binary2bit => QuantKind::Binary2bit,
            QuantizationArg::Binary1p5bit => QuantKind::Binary15bit,
            QuantizationArg::Turbo1bit => QuantKind::Turbo1bit,
            QuantizationArg::Turbo1p5bit => QuantKind::Turbo15bit,
            QuantizationArg::Turbo2bit => QuantKind::Turbo2bit,
            QuantizationArg::Turbo4bit => QuantKind::Turbo4bit,
            QuantizationArg::Scalar => QuantKind::Scalar,
            QuantizationArg::ProductX4 => QuantKind::ProductX4,
            QuantizationArg::ProductX8 => QuantKind::ProductX8,
            QuantizationArg::ProductX16 => QuantKind::ProductX16,
            QuantizationArg::ProductX32 => QuantKind::ProductX32,
            QuantizationArg::ProductX64 => QuantKind::ProductX64,
        }
    }
}

/// Memory placement of a component's data (Qdrant 1.19+). Data is always
/// persisted on disk; this only controls how it is held in RAM.
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum MemoryArg {
    /// Not pre-loaded from disk; cached with usage.
    Cold,
    /// Pre-loaded into disk-cache RAM on start, may be evicted under pressure.
    Cached,
    /// Loaded in RAM and never evicted. Unsupported for dense vectors and payload storage.
    Pinned,
}

impl From<MemoryArg> for MemoryKind {
    fn from(arg: MemoryArg) -> Self {
        match arg {
            MemoryArg::Cold => MemoryKind::Cold,
            MemoryArg::Cached => MemoryKind::Cached,
            MemoryArg::Pinned => MemoryKind::Pinned,
        }
    }
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
    /// Optional subcommand. When omitted, runs the legacy flag-driven benchmark.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Qdrant service URI
    #[clap(long, default_value = "http://localhost:6334", global = true)]
    pub uri: Vec<String>,

    /// Source of data to upload - fbin file. Random if not specified
    #[clap(long)]
    pub fbin: Option<String>,

    /// Number of points to upload. When dataset sources are used and omitted,
    /// the full dataset is uploaded (capped by the smallest source).
    #[clap(short, long, value_parser = parse_number, global = true)]
    pub num_vectors: Option<usize>,

    /// Number of named dense vectors per point
    #[clap(long, default_value_t = 1, value_parser = parse_number)]
    pub vectors_per_point: usize,

    #[clap(short, long, default_value_t = 0, value_parser = parse_number, global = true)]
    pub offset: usize,

    /// If set, will randomly upsert/override vector ids within range [offset, max_id)
    #[clap(short, long, value_parser = parse_number, global = true)]
    pub max_id: Option<usize>,

    /// Number of dimensions in each dense vector or max dimension for sparse vectors
    #[clap(short, long, default_value_t = 128, value_parser = parse_number)]
    pub dim: usize,

    /// Number of worker threads to use
    #[clap(short, long, default_value_t = 2, value_parser = parse_number, global = true)]
    pub threads: usize,

    /// Number of parallel requests to send (ignored when --rps is set)
    #[clap(short, long, default_value_t = 2, value_parser = parse_number, global = true)]
    pub parallel: usize,

    /// Target requests per second. When set, requests are sent at a fixed rate
    /// regardless of how many are currently in-flight (simulates real user traffic).
    /// This overrides --parallel for concurrency control.
    #[clap(long, value_name = "RATE", global = true)]
    pub rps: Option<f64>,

    /// Number of connections to open from the client to the server
    #[clap(short, long, default_value_t = 1, value_parser = parse_number, global = true)]
    pub connections: usize,

    /// Batch size for updates, in number of points. [default=100]
    #[clap(short, long, value_name = "POINTS", default_value_t = 100, value_parser = parse_number, global = true)]
    pub batch_size: usize,

    /// Batch size for searches, in number of queries per batch.
    #[clap(long, value_name = "QUERIES", default_value_t = 1, value_parser = parse_number, global = true)]
    pub search_batch_size: usize,

    /// Throttle updates and searches, in batches/searches per second. [default=no throttling]
    #[clap(long, short = 'T', value_name = "RPS", global = true)]
    pub throttle: Option<f32>,

    /// Skip creating a collection
    #[clap(long, default_value_t = false, global = true)]
    pub skip_create: bool,

    /// Create if not exists. Avoid re-creating collection
    #[clap(long, default_value_t = false, global = true)]
    pub create_if_missing: bool,

    /// Skip wait until collection is indexed after upload
    #[clap(long, default_value_t = false, global = true)]
    pub skip_wait_index: bool,

    /// Skip uploading new points
    #[clap(long, default_value_t = false, global = true)]
    pub skip_upload: bool,

    /// Skip setting up collections.
    /// Implies --skip-create --skip-upload --skip-wait-index
    #[clap(long, default_value_t = false, global = true)]
    pub skip_setup: bool,

    /// Perform search
    #[clap(long, default_value_t = false)]
    pub search: bool,

    /// Perform search without approximation
    #[clap(long, default_value_t = false, global = true)]
    pub search_exact: bool,

    /// Prefetch search
    #[clap(long, value_parser = parse_number, global = true)]
    pub prefetch: Option<usize>,

    /// Perform scroll
    #[clap(long, default_value_t = false)]
    pub scroll: bool,

    /// Search limit
    #[clap(long, default_value_t = 10, value_parser = parse_number, global = true)]
    pub search_limit: usize,

    /// Write the benchmark results document (config + every phase) to this path
    #[clap(long, global = true)]
    pub json: Option<String>,

    /// Number of 9 digits to show in p99* results
    #[clap(long, long, default_value_t = 2, value_parser = parse_number, global = true)]
    pub p9: usize,

    /// Name of the collection to use
    #[clap(long, default_value = "benchmark")]
    pub collection_name: String,

    /// Distance function used for comparing vectors
    #[clap(long, default_value = "Cosine")]
    pub distance: String,

    /// Vector datatypes (Uint8, Float16, Float32, Turbo4)
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

    /// Memory placement of the payload storage (supersedes --on-disk-payload)
    #[clap(long, value_name = "PLACEMENT")]
    pub memory_payload: Option<MemoryArg>,

    /// Memory placement of the payload field indices (supersedes --on-disk-payload-index)
    #[clap(long, value_name = "PLACEMENT")]
    pub memory_payload_index: Option<MemoryArg>,

    /// Memory placement of the HNSW graph and sparse index (supersedes --on-disk-index)
    #[clap(long, value_name = "PLACEMENT")]
    pub memory_index: Option<MemoryArg>,

    /// Memory placement of the vector storage (supersedes --on-disk-vectors)
    #[clap(long, value_name = "PLACEMENT")]
    pub memory_vectors: Option<MemoryArg>,

    /// Memory placement of the quantized vectors (supersedes --quantization-in-ram)
    #[clap(long, value_name = "PLACEMENT")]
    pub memory_quantization: Option<MemoryArg>,

    /// Log requests if the take longer than this
    #[clap(long, default_value_t = 0.1, global = true)]
    pub timing_threshold: f64,

    /// Use UUIDs instead of sequential ids
    #[clap(long, default_value_t = false)]
    pub uuids: bool,

    /// Skip field indices creation if payloads are not empty
    #[clap(long, default_value_t = false, global = true)]
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
    #[clap(long, default_value_t = false, global = true)]
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
    #[clap(long, value_parser = parse_number, global = true)]
    pub search_hnsw_ef: Option<usize>,

    /// Store HNSW links inline with vectors (hnsw_config.inline_storage)
    #[clap(long, default_value_t = false)]
    pub hnsw_inline_storage: bool,

    /// Whether to request payload in search results
    #[clap(long, default_value_t = false, global = true)]
    pub search_with_payload: bool,

    /// Whether to request vectors in search results
    #[clap(long, default_value_t = false, global = true)]
    pub search_with_vectors: bool,

    /// Wait on upsert
    #[clap(long, default_value_t = false, global = true)]
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
    #[clap(long, global = true)]
    pub write_ordering: Option<WriteOrdering>,

    /// Read consistency parameter to use for all read requests
    #[clap(long, global = true)]
    pub read_consistency: Option<ReadConsistency>,

    /// Timeout for requests in seconds (applied as both the client channel deadline and the server-side request timeout where supported).
    #[clap(long, value_parser = parse_number, global = true)]
    pub timeout: Option<usize>,

    /// Number of retries for each URI on error, 0 for no retries
    #[clap(long = "retry", default_value_t = 0, value_parser = parse_number, global = true)]
    pub retries: usize,

    /// Number of seconds between each retry.
    #[clap(long, default_value_t = 0.0, value_name = "SECONDS", global = true)]
    pub retry_interval: f32,

    /// Keep going on search error
    #[clap(long, default_value_t = false, global = true)]
    pub ignore_errors: bool,

    #[clap(long)]
    pub quantization: Option<QuantizationArg>,

    /// Keep quantized vectors in memory
    #[clap(long)]
    pub quantization_in_ram: Option<bool>,

    /// Enable quantization re-score during search
    #[clap(long, global = true)]
    pub quantization_rescore: Option<bool>,

    /// Quantization oversampling factor
    #[clap(long, global = true)]
    pub quantization_oversampling: Option<f64>,

    /// Delay between requests in milliseconds
    #[clap(long, value_parser = parse_number, global = true)]
    pub delay: Option<usize>,

    /// Skip un-indexed segments during search
    #[clap(long, global = true)]
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

    /// Create sparse vectors with the IDF modifier (BM25-style scoring).
    /// Required by --search-idf-corpus.
    #[clap(long, default_value_t = false)]
    pub sparse_idf: bool,

    /// Path to the jsonl file to save update timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long, global = true)]
    pub jsonl_updates: Option<String>,

    /// Path to the jsonl file to save search timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long, global = true)]
    pub jsonl_searches: Option<String>,

    /// Path to the jsonl file to save rps timings
    /// TIP: Use `qdrant/mri` to visualize the timings
    #[clap(long, global = true)]
    pub jsonl_rps: Option<String>,

    /// Use timestamp instead of relative time in jsonl
    /// Default is relative time
    #[clap(long, global = true)]
    pub absolute_time: Option<bool>,

    /// Use custom sharding for collection and upsert points to the specified sharding key
    #[clap(long)]
    pub shard_key: Option<String>,

    /// Use tenant optimization for field index.
    #[clap(long)]
    pub tenants: Option<bool>,

    /// Use a custom UUID as filter when searching.
    #[clap(long, global = true)]
    pub uuid_query: Option<String>,

    /// Bench for search quality / accurracy too.
    #[clap(long, global = true)]
    pub search_quality: bool,

    /// Set a custom full-scan threshold.
    #[clap(long)]
    pub full_scan_threshold: Option<usize>,

    /// Compute sparse-vector IDF statistics over the points matching the query
    /// filter instead of the whole collection. Requires a filter to be generated
    /// (e.g. `-k`), and sparse vectors created with the IDF modifier.
    #[clap(long, default_value_t = false, global = true)]
    pub search_idf_corpus: bool,
}

/// Subcommands. Absent ⇒ legacy flag-driven benchmark.
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Upload data described by a YAML collection-shape config.
    ///
    /// The YAML defines the *shape* of the data (collection params + how
    /// each field is generated); the runtime flags (`-n`, `-b`, `-p`, `-t`,
    /// `--uri`, …) still control *how* it is uploaded. Use `--example name`
    /// for a built-in config (`bfb examples` lists them) or `--file` for a
    /// custom YAML file.
    Upload(UploadArgs),

    /// Run searches described by a YAML search-request config.
    ///
    /// The YAML defines the *shape* of search requests (which vectors to
    /// query, optional payload filters); the runtime flags (`-n`, `--parallel`,
    /// `--search-batch-size`, `--uri`, …) still control *how* the benchmark
    /// runs. Use `--example name` or `--file path`.
    Search(SearchArgs),

    /// Run a scroll workload against an existing collection.
    ///
    /// The same workload as the legacy `--scroll` flag, runnable as a phase of
    /// its own so it can be timed and reported independently. The YAML
    /// defines the *shape* of scroll requests (which payload filters); the
    /// runtime flags (`-n`, `--parallel`, `--search-limit`, …) still control
    /// *how* the benchmark runs. Use `--example name` or `--file path`.
    Scroll(ScrollArgs),

    /// List built-in YAML examples, or print one to stdout.
    ///
    /// `bfb examples` lists names for `--example`. `bfb examples <name>` prints
    /// the YAML so it can be saved and edited as a `--file`.
    Examples(ExamplesArgs),

    /// Print the upload-config file schema: every option, its type, default,
    /// and allowed values, as an annotated YAML reference.
    Schema,

    /// Replace this binary with the latest GitHub release (or a chosen `--tag`).
    ///
    /// Downloads the prebuilt `bfb-<target>.tar.gz` for this platform from
    /// https://github.com/qdrant/bfb/releases, verifies its checksum, and
    /// atomically swaps it in place of the running executable.
    SelfUpdate(crate::self_update::SelfUpdateArgs),
}

/// `--file` or `--example` (exactly one required). Flattened into upload /
/// search / scroll.
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = false)]
pub struct ConfigArgs {
    /// Path to a YAML config file
    #[clap(long)]
    pub file: Option<String>,

    /// Built-in example name (`bfb examples` lists them)
    #[clap(long, value_name = "NAME")]
    pub example: Option<String>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct SearchArgs {
    #[clap(flatten)]
    pub config: ConfigArgs,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ScrollArgs {
    #[clap(flatten)]
    pub config: ConfigArgs,
}

#[derive(clap::Args, Debug, Clone)]
pub struct UploadArgs {
    #[clap(flatten)]
    pub config: ConfigArgs,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ExamplesArgs {
    /// Print this built-in example's YAML to stdout. Omit to list names.
    pub name: Option<String>,
}

impl Args {
    pub fn is_uint8_datatype(&self) -> bool {
        self.datatype
            .as_ref()
            .is_some_and(|x| x == qdrant::Datatype::Uint8.as_str_name())
    }

    /// Effective point count for legacy mode (no dataset sources).
    pub fn num_vectors_or_default(&self) -> usize {
        self.num_vectors.unwrap_or(100_000)
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
