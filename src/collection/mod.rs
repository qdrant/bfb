//! Collection lifecycle: (re)creation — from CLI flags or a YAML config —
//! waiting for indexing, building payload field indices, and patching settings
//! on a live collection.

mod from_args;
mod from_config;

pub use from_args::recreate_collection;
pub use from_config::{field_index_specs_from_config, recreate_collection_from_config};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use qdrant_client::qdrant::{
    CollectionInfo, CollectionStatus, CreateFieldIndexCollection,
    CreateFieldIndexCollectionBuilder, DeleteFieldIndexCollectionBuilder, HnswConfigDiffBuilder,
    OptimizersConfigDiffBuilder, UpdateCollectionBuilder,
};
use tokio::time::sleep;

use crate::args::Args;
use crate::client::random_client;
use crate::config::update::UpdateConfig;
use crate::results::{CreateFieldIndexPhase, FieldIndexTiming, UpdateCollectionPhase};

/// One field index to create: the built request plus the labels used to report it.
pub struct FieldIndexSpec {
    pub field: String,
    /// Payload type name, e.g. `keyword`, `integer`.
    pub kind: &'static str,
    pub request: CreateFieldIndexCollection,
}

impl FieldIndexSpec {
    pub fn new(
        field: &str,
        kind: &'static str,
        builder: CreateFieldIndexCollectionBuilder,
    ) -> Self {
        FieldIndexSpec {
            field: field.to_string(),
            kind,
            // `wait(true)` is what makes the elapsed time meaningful: without it
            // the server returns before the index is built.
            request: builder.wait(true).build(),
        }
    }
}

/// Create field indices and time each one.
///
/// Called both while creating an empty collection (where the timings are
/// near-zero and ignored) and, via `bfb create-field-index`, on a populated
/// collection — which is the measurement the payload-indexing benchmarks want.
pub async fn create_field_indices(
    client: &qdrant_client::Qdrant,
    specs: Vec<FieldIndexSpec>,
) -> Result<CreateFieldIndexPhase> {
    let started = Instant::now();
    let mut fields = Vec::with_capacity(specs.len());

    for spec in specs {
        let field_started = Instant::now();
        let response = client.create_field_index(spec.request).await?;
        fields.push(FieldIndexTiming {
            field: spec.field,
            kind: spec.kind.to_string(),
            duration_secs: field_started.elapsed().as_secs_f64(),
            server_secs: response.time,
        });
    }

    Ok(CreateFieldIndexPhase {
        duration_secs: started.elapsed().as_secs_f64(),
        fields,
    })
}

/// Drop the given field indices, so they can be rebuilt and re-measured on the
/// same data without re-uploading it.
pub async fn drop_field_indices(
    client: &qdrant_client::Qdrant,
    collection: &str,
    fields: &[String],
) -> Result<()> {
    for field in fields {
        client
            .delete_field_index(
                DeleteFieldIndexCollectionBuilder::new(collection, field).wait(true),
            )
            .await?;
        println!("Dropped field index: {field}");
    }
    Ok(())
}

/// Wait until the collection status is stably `Green`; returns the wait time
/// in seconds.
pub async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let client = random_client(args)?;
    let start = Instant::now();
    let mut seen = 0;
    let mut last_report = Instant::now();
    loop {
        if stopped.load(Ordering::Relaxed) {
            return Ok(0.0);
        }
        let info = client.collection_info(&args.collection_name).await?;
        let Some(result) = info.result else {
            bail!(
                "collection_info returned no result for {}",
                args.collection_name
            );
        };
        if result.status == CollectionStatus::Green as i32 {
            seen += 1;
            if seen == GREEN_CONFIRMATIONS {
                break;
            }
        } else {
            seen = 0;
            if last_report.elapsed() >= INDEX_PROGRESS_EVERY {
                report_index_progress(&result, start.elapsed());
                last_report = Instant::now();
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
    Ok(start.elapsed().as_secs_f64())
}

/// Consecutive green polls required before the collection counts as indexed. The
/// optimizer is not scheduled synchronously with the request that triggers it, so a
/// single green reading can just mean it has not started yet.
const GREEN_CONFIRMATIONS: u32 = 3;

/// How often to print a line while waiting, so a long rebuild does not look like a hang.
const INDEX_PROGRESS_EVERY: Duration = Duration::from_secs(5);

fn report_index_progress(info: &CollectionInfo, elapsed: Duration) {
    let status = match CollectionStatus::try_from(info.status) {
        Ok(CollectionStatus::Yellow) => "yellow",
        Ok(CollectionStatus::Grey) => "grey",
        Ok(CollectionStatus::Red) => "red",
        _ => "unknown",
    };
    let indexed = info.indexed_vectors_count.unwrap_or(0);
    let total = info.points_count.unwrap_or(0);
    let pct = if total > 0 {
        format!(" ({:.1}%)", 100.0 * indexed as f64 / total as f64)
    } else {
        String::new()
    };
    println!(
        "  indexing: status={status}, {indexed}/{total} vectors{pct}, {:.0}s elapsed",
        elapsed.as_secs_f64()
    );
}

/// Patch collection settings on a live collection.
///
/// Only what the config declares is sent, so this can lower
/// `indexing_threshold` to start indexing on demand, or change
/// `max_segment_size` to trigger a merge, without disturbing anything else.
pub async fn update_collection(
    args: &Args,
    config: &UpdateConfig,
) -> Result<UpdateCollectionPhase> {
    let client = random_client(args)?;
    let mut changed = Vec::new();

    let mut optimizers = OptimizersConfigDiffBuilder::default();
    let mut optimizers_changed = false;
    if let Some(patch) = &config.collection.optimizers {
        if let Some(threshold) = patch.indexing_threshold {
            optimizers = optimizers.indexing_threshold(threshold);
            optimizers_changed = true;
            changed.push(format!("optimizers.indexing_threshold={threshold}"));
        }
        if let Some(size) = patch.max_segment_size {
            optimizers = optimizers.max_segment_size(size);
            optimizers_changed = true;
            changed.push(format!("optimizers.max_segment_size={size}"));
        }
        if let Some(threshold) = patch.memmap_threshold {
            optimizers = optimizers.memmap_threshold(threshold);
            optimizers_changed = true;
            changed.push(format!("optimizers.memmap_threshold={threshold}"));
        }
        if let Some(segments) = patch.default_segment_number {
            optimizers = optimizers.default_segment_number(segments);
            optimizers_changed = true;
            changed.push(format!("optimizers.default_segment_number={segments}"));
        }
    }

    let mut hnsw = HnswConfigDiffBuilder::default();
    let mut hnsw_changed = false;
    if let Some(patch) = &config.collection.hnsw {
        if let Some(m) = patch.m {
            hnsw = hnsw.m(m);
            hnsw_changed = true;
            changed.push(format!("hnsw.m={m}"));
        }
        if let Some(ef) = patch.ef_construct {
            hnsw = hnsw.ef_construct(ef);
            hnsw_changed = true;
            changed.push(format!("hnsw.ef_construct={ef}"));
        }
        if let Some(payload_m) = patch.payload_m {
            hnsw = hnsw.payload_m(payload_m);
            hnsw_changed = true;
            changed.push(format!("hnsw.payload_m={payload_m}"));
        }
        if let Some(on_disk) = patch.on_disk {
            hnsw = hnsw.on_disk(on_disk);
            hnsw_changed = true;
            changed.push(format!("hnsw.on_disk={on_disk}"));
        }
    }

    // `validate()` already rejected an empty patch, so `changed` is non-empty.
    // Each section is only attached when it actually changed: an empty diff would
    // be a no-op at best, and could still nudge the server to re-evaluate work
    // the patch never asked for.
    let mut builder = UpdateCollectionBuilder::new(args.collection_name.clone());
    if optimizers_changed {
        builder = builder.optimizers_config(optimizers);
    }
    if hnsw_changed {
        builder = builder.hnsw_config(hnsw);
    }

    let started = Instant::now();
    let response = client.update_collection(builder).await?;
    let duration_secs = started.elapsed().as_secs_f64();

    println!("Updated collection: {}", changed.join(", "));
    println!("Server reported changes: {}", response.result);

    Ok(UpdateCollectionPhase {
        duration_secs,
        server_secs: response.time,
        changed,
        applied: response.result,
    })
}
