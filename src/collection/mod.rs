//! Collection lifecycle: (re)creation — from CLI flags or a YAML config —
//! waiting for indexing, and building payload field indices.

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
    CreateFieldIndexCollectionBuilder, DeleteFieldIndexCollectionBuilder,
};
use tokio::time::sleep;

use crate::args::Args;
use crate::client::random_client;
use crate::results::{CreateFieldIndexPhase, FieldIndexTiming};

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
