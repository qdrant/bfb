//! Build [`QdrantServerless`] clients from shared BFB [`Args`].

use std::time::Duration;

use anyhow::Result;
use qdrant_client::serverless::QdrantServerless;
use rand::RngExt;
use rand::prelude::SliceRandom;
use tracing::warn;

use crate::args::Args;

fn choose_owned<T>(mut items: Vec<T>) -> T {
    let mut rng = rand::rng();
    let id = rng.random_range(0..items.len());
    items.swap_remove(id)
}

/// One serverless client per (`uri` × `connections`) pair, matching regular BFB.
pub fn create_clients(args: &Args) -> Result<Vec<QdrantServerless>> {
    let api_key = std::env::var("QDRANT_API_KEY").ok();
    let mut clients = Vec::new();

    for _ in 0..args.connections {
        for uri in &args.uri {
            let mut builder = QdrantServerless::from_url(uri);
            if let Some(timeout) = args.timeout {
                let channel_timeout = Duration::from_secs(timeout as u64 + 5);
                builder = builder
                    .timeout(channel_timeout)
                    .connect_timeout(channel_timeout);
            }
            if let Some(api_key) = &api_key {
                builder = builder.api_key(api_key.as_str());
            }
            clients.push(builder.build()?);
        }
    }
    Ok(clients)
}

pub fn random_client(args: &Args) -> Result<QdrantServerless> {
    Ok(choose_owned(create_clients(args)?))
}

/// Try the request on every client in random order, retrying `args.retries`
/// times with `args.retry_interval` between rounds.
pub async fn retry_with_clients<
    'a,
    R,
    T: std::future::Future<Output = Result<R, qdrant_client::QdrantError>>,
>(
    clients: &'a [QdrantServerless],
    args: &Args,
    mut call: impl FnMut(&'a QdrantServerless) -> T,
) -> anyhow::Result<R> {
    let mut rng = rand::rng();
    let mut permutation = (0..clients.len()).collect::<Vec<_>>();
    let mut previous_err: Option<anyhow::Error> = None;

    for attempt in 0..=args.retries {
        permutation.shuffle(&mut rng);
        for client_id in &permutation {
            let client = &clients[*client_id];
            match call(client).await {
                Ok(v) => return Ok(v),
                Err(err) => previous_err = Some(err.into()),
            }
        }

        if attempt < args.retries {
            if let Some(err) = &previous_err {
                warn!("Request failed at attempt {}: {err}", attempt + 1);
            }
            tokio::time::sleep(Duration::from_secs_f32(args.retry_interval.max(0.0))).await;
        }
    }

    Err(previous_err.unwrap_or_else(|| anyhow::anyhow!("No clients")))
}
