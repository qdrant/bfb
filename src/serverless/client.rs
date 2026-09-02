//! Build [`QdrantServerless`] clients from shared BFB [`Args`].
//!
//! Reuses the regular client configuration (`--uri` × `--connections`,
//! `--timeout`, `QDRANT_API_KEY`) so both modes are configured the same way.

use anyhow::Result;
use qdrant_client::serverless::QdrantServerless;

use crate::args::Args;
use crate::client::get_config;

/// One serverless client per (`uri` × `connections`) pair, matching regular BFB.
pub fn create_clients(args: &Args) -> Result<Vec<QdrantServerless>> {
    get_config(args)
        .into_iter()
        .map(|config| Ok(QdrantServerless::new(config)?))
        .collect()
}

/// A single client for one-off administrative calls (`clear`, listing).
pub fn single_client(args: &Args) -> Result<QdrantServerless> {
    let config = get_config(args)
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no --uri given"))?;
    Ok(QdrantServerless::new(config)?)
}
