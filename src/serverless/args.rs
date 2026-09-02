//! CLI for `bfb serverless {upload,clear,query}`.

use clap::{Args as ClapArgs, Subcommand, ValueEnum};

use super::distribution::Distribution;
use crate::args::ConfigArgs;

/// `bfb serverless` — multi-collection benchmarks against Qdrant Serverless.
#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessArgs {
    #[command(subcommand)]
    pub command: ServerlessCommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ServerlessCommand {
    /// Upload points across a range of collections (created lazily on first use).
    Upload(ServerlessUploadArgs),

    /// Delete every collection whose name starts with `--collection-prefix`.
    Clear(ServerlessClearArgs),

    /// List collections and their point counts.
    List(ServerlessListArgs),

    /// Run queries routed across existing collections matching the prefix.
    Query(ServerlessQueryArgs),
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessUploadArgs {
    /// Prefix shared by every collection name (`benchmark-` → `benchmark-0` …).
    #[clap(long)]
    pub collection_prefix: String,

    /// How many collections to spread points across.
    #[clap(long, value_parser = crate::args::parse_number)]
    pub collections_count: usize,

    /// How points are allocated across collections.
    #[clap(long, value_enum, default_value_t = DistributionArg::Uniform)]
    pub distribution: DistributionArg,

    /// Total number of points to upload across all collections.
    /// Falls back to the global `-n` / `--num-vectors` when omitted.
    #[clap(long, value_parser = crate::args::parse_number)]
    pub total_points: Option<usize>,

    /// Upload-shape YAML: `--file <path>` or `--example <name>` (same schema
    /// as `bfb upload`).
    #[clap(flatten)]
    pub config: ConfigArgs,
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessClearArgs {
    /// Delete every collection whose name starts with this prefix.
    #[clap(long)]
    pub collection_prefix: String,
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessListArgs {
    /// Only list collections whose name starts with this prefix.
    /// Omit to list every collection in the space.
    #[clap(long, default_value = "")]
    pub collection_prefix: String,
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessQueryArgs {
    /// Query every existing collection whose name starts with this prefix.
    #[clap(long)]
    pub collection_prefix: String,

    /// How queries are routed across matching collections.
    #[clap(long, value_enum, default_value_t = DistributionArg::Uniform)]
    pub distribution: DistributionArg,

    /// Optional search-shape YAML (same schema as `bfb search`). When omitted,
    /// one random dense or sparse query template is derived from a matching
    /// collection's config.
    #[clap(flatten)]
    pub config: OptionalConfigArgs,
}

/// `--file` or `--example`, both optional (at most one).
#[derive(ClapArgs, Debug, Clone)]
#[group(multiple = false)]
pub struct OptionalConfigArgs {
    /// Path to a YAML config file
    #[clap(long)]
    pub file: Option<String>,

    /// Built-in example name (`bfb examples` lists them)
    #[clap(long, value_name = "NAME")]
    pub example: Option<String>,
}

impl OptionalConfigArgs {
    pub fn is_some(&self) -> bool {
        self.file.is_some() || self.example.is_some()
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum DistributionArg {
    Uniform,
    Zipf,
}

impl From<DistributionArg> for Distribution {
    fn from(value: DistributionArg) -> Self {
        match value {
            DistributionArg::Uniform => Distribution::Uniform,
            DistributionArg::Zipf => Distribution::Zipf,
        }
    }
}
