//! Collection lifecycle: (re)creation — from CLI flags or a YAML config — and
//! waiting for indexing to finish.

mod from_args;
mod from_config;

pub use from_args::recreate_collection;
pub use from_config::recreate_collection_from_config;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::qdrant::CollectionStatus;
use tokio::time::sleep;

use crate::args::Args;
use crate::client::random_client;

/// Wait until the collection status is stably `Green`; returns the wait time
/// in seconds.
pub async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let client = random_client(args)?;
    let start = std::time::Instant::now();
    let mut seen = 0;
    loop {
        if stopped.load(Ordering::Relaxed) {
            return Ok(0.0);
        }
        sleep(Duration::from_secs(1)).await;
        let info = client.collection_info(&args.collection_name).await?;
        if info.result.unwrap().status == CollectionStatus::Green as i32 {
            seen += 1;
            if seen == 3 {
                break;
            }
        } else {
            seen = 1;
        }
    }
    Ok(start.elapsed().as_secs_f64())
}
