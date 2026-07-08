use std::time::Duration;

use anyhow::{Error, Result};
use qdrant_client::config::QdrantConfig;
use qdrant_client::{Qdrant, QdrantError};
use rand::RngExt;
use rand::prelude::SliceRandom;
use tracing::warn;

use crate::args::Args;

fn choose_owned<T>(mut items: Vec<T>) -> T {
    let mut rng = rand::rng();
    // Get random id
    let id = rng.random_range(0..items.len());
    // Remove item from vector
    items.swap_remove(id)
}

pub fn get_config(args: &Args) -> Vec<QdrantConfig> {
    let mut configs = Vec::new();

    for _i in 0..args.connections {
        for uri in &args.uri {
            let mut config = QdrantConfig::from_url(uri);
            let api_key = std::env::var("QDRANT_API_KEY").ok();

            if let Some(timeout) = args.timeout {
                // Give the channel a small cushion over the server-side request timeout
                // so the server's structured timeout error wins the race over a transport deadline.
                let channel_timeout = Duration::from_secs(timeout as u64 + 5);
                config.set_timeout(channel_timeout);
                config.set_connect_timeout(channel_timeout);
            }

            if let Some(api_key) = api_key {
                config.set_api_key(&api_key);
            }
            configs.push(config);
        }
    }
    configs
}

pub fn random_client(args: &Args) -> Result<Qdrant> {
    Ok(Qdrant::new(choose_owned(get_config(args)))?)
}

pub fn create_clients(args: &Args) -> Result<Vec<Qdrant>> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(Qdrant::new(config)?);
    }
    Ok(clients)
}

/// Try the request on every client in random order, retrying `args.retries`
/// times with `args.retry_interval` between rounds.
pub async fn retry_with_clients<'a, R, T: std::future::Future<Output = Result<R, QdrantError>>>(
    clients: &'a [Qdrant],
    args: &Args,
    mut call: impl FnMut(&'a Qdrant) -> T,
) -> anyhow::Result<R> {
    let mut rng = rand::rng();
    let mut permutation = (0..clients.len()).collect::<Vec<_>>();
    let mut previous_err: Option<Error> = None;

    for attempt in 0..=args.retries {
        permutation.shuffle(&mut rng);
        for client_id in &permutation {
            let client = clients.get(*client_id).unwrap();

            let resp = call(client).await.map_err(|i| i.into());

            match resp {
                Ok(_) => {
                    return resp;
                }
                Err(err) => {
                    previous_err = Some(err);
                }
            }
        }

        let is_last = attempt >= args.retries;
        if !is_last {
            if let Some(err) = &previous_err {
                warn!("Request failed at attempt {}: {err}", attempt + 1);
            }

            tokio::time::sleep(Duration::from_secs_f32(args.retry_interval.max(0.0))).await;
        }
    }

    Err(previous_err.unwrap_or_else(|| anyhow::anyhow!("No clients")))
}
