use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScrollPointsBuilder;

use crate::args::Args;
use crate::client::create_clients;
use crate::common::UUID_PAYLOAD_KEY;
use crate::scroll::ScrollProcessor;
use crate::search::SearchProcessor;
use crate::stats::process;
use crate::structured_vectors::StructuredVectorGenerator;

pub async fn search(
    args: &Args,
    generator: Option<Arc<StructuredVectorGenerator>>,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let clients = create_clients(args)?;
    let uuids = get_uuids(args, &clients[0]).await?;

    let searcher = SearchProcessor::new(args.clone(), generator, stopped.clone(), clients, uuids);
    process(args, stopped, searcher).await
}

pub async fn scroll(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let clients = create_clients(args)?;
    let uuids = get_uuids(args, &clients[0]).await?;

    let scroller = ScrollProcessor::new(args.clone(), stopped.clone(), clients, uuids);
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
    let res = client
        .scroll(
            ScrollPointsBuilder::new(&args.collection_name)
                .with_payload(true)
                .limit(args.num_vectors as u32),
        )
        .await?;
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
