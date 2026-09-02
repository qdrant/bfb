//! CLI for `bfb serverless {upload,clear,query}`.

use clap::{Args as ClapArgs, Subcommand, ValueEnum};

use super::distribution::Distribution;

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

    /// Path to a YAML upload-shape config (`bfb upload --file` schema).
    /// Alias of `--file` for the Notion CLI wording.
    #[clap(long = "config-file", visible_alias = "file", value_name = "PATH")]
    pub config_file: String,
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ServerlessClearArgs {
    /// Delete every collection whose name starts with this prefix.
    #[clap(long)]
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

    /// Optional YAML search-shape config. When omitted, vector shape is read
    /// from an existing collection's serverless config.
    #[clap(long = "config-file", visible_alias = "file", value_name = "PATH")]
    pub config_file: Option<String>,
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
