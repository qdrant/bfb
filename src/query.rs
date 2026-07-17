use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::qdrant::shard_key::Key;

use crate::args::Args;
use crate::client::create_clients;
use crate::config::scroll::ScrollConfig;
use crate::config::search::SearchConfig;
use crate::generators::random::UUID_PAYLOAD_KEY;
use crate::results::QueryPhase;
use crate::scroll::ScrollProcessor;
use crate::search::ConfigSearchProcessor;
use crate::search::SearchProcessor;
use crate::stats::process;

pub async fn search(args: &Args, stopped: Arc<AtomicBool>) -> Result<QueryPhase> {
    let clients = create_clients(args)?;
    let uuids = get_uuids(args, &clients[0]).await?;

    let searcher = SearchProcessor::new(args.clone(), stopped.clone(), clients, uuids);
    process(args, stopped, searcher).await
}

/// YAML-config-driven search (`bfb search --file config.yaml`).
pub async fn search_with_config(
    args: &Args,
    config: &SearchConfig,
    stopped: Arc<AtomicBool>,
) -> Result<QueryPhase> {
    let clients = create_clients(args)?;
    let searcher = ConfigSearchProcessor::new(args.clone(), config, stopped.clone(), clients)?;
    process(args, stopped, searcher).await
}

pub async fn scroll(args: &Args, stopped: Arc<AtomicBool>) -> Result<QueryPhase> {
    let clients = create_clients(args)?;
    let uuids = get_uuids(args, &clients[0]).await?;

    let scroller = ScrollProcessor::new(args.clone(), stopped.clone(), clients, uuids);
    process(args, stopped, scroller).await
}

/// YAML-config-driven scroll (`bfb scroll --file config.yaml`).
pub async fn scroll_with_config(
    args: &Args,
    config: &ScrollConfig,
    stopped: Arc<AtomicBool>,
) -> Result<QueryPhase> {
    let clients = create_clients(args)?;
    let scroller = ScrollProcessor::from_config(args.clone(), config, stopped.clone(), clients);
    process(args, stopped, scroller).await
}

/// If we want to retrieve points by UUIDs, we need to know about the existing UUIDs.
/// Here we decide which UUIDs we want to use for searching, based on the user's preference.
async fn get_uuids(args: &Args, client: &Qdrant) -> Result<Vec<String>> {
    // Only use the UUID the user specified
    if let Some(uuid_query) = &args.uuid_query {
        return Ok(vec![uuid_query.to_string()]);
    }

    if !args.uuid_payloads {
        return Ok(vec![]);
    }

    // Retrieve existing UUIDs
    let mut scroll_builder = ScrollPointsBuilder::new(&args.collection_name)
        .with_payload(true)
        .limit(args.num_vectors_or_default() as u32);
    if let Some(shard_key) = &args.shard_key {
        scroll_builder = scroll_builder.shard_key_selector(vec![Key::Keyword(shard_key.clone())]);
    }
    if let Some(timeout) = args.timeout {
        scroll_builder = scroll_builder.timeout(timeout as u64);
    }
    let res = client.scroll(scroll_builder).await?;
    let uuids: Vec<_> = res
        .result
        .iter()
        .filter_map(|i| {
            i.payload
                .get(UUID_PAYLOAD_KEY)
                .and_then(|j| j.as_str().map(|i| i.to_string()))
        })
        .collect();
    let uuids_count = uuids.len();
    let unique: HashSet<_> = uuids.into_iter().collect();
    if unique.len() != uuids_count {
        println!("Set of uuids not unique!");
    }

    // Make order random to not request the first point by its UUID.
    Ok(unique.into_iter().collect())
}
