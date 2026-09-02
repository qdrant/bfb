//! Serverless benchmarking mode.
//!
//! Unlike the regular single-collection workflow, serverless mode spreads work
//! across a *range* of collections (one per tenant). Collections are created
//! lazily on first upsert; the registry tracks which existed before the run
//! and which were created during upload.
//!
//! ```text
//! bfb serverless upload --collection-prefix benchmark- --collections-count 100 \
//!     --distribution uniform --total-points 10M --example serverless-upload
//! bfb serverless clear  --collection-prefix benchmark-
//! bfb serverless query  --collection-prefix benchmark- --distribution zipf -n 10k
//! ```

mod args;
mod clear;
mod client;
mod collections;
mod convert;
mod distribution;
mod query;
mod upload;

pub use args::{ServerlessArgs, ServerlessCommand};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::args::Args;

/// Dispatch a `bfb serverless …` subcommand.
pub async fn run(args: Args, serverless: ServerlessArgs, stopped: Arc<AtomicBool>) -> Result<()> {
    match serverless.command {
        ServerlessCommand::Upload(upload_args) => upload::run(&args, upload_args, stopped).await,
        ServerlessCommand::Clear(clear_args) => clear::run(&args, clear_args).await,
        ServerlessCommand::Query(query_args) => query::run(&args, query_args, stopped).await,
    }
}
