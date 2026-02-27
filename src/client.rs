use std::time::Duration;

use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::config::QdrantConfig;
use rand::RngExt;

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
                config.set_timeout(Duration::from_secs(timeout as u64));
                config.set_connect_timeout(Duration::from_secs(timeout as u64));
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
