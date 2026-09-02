//! `bfb serverless upload` — spread points across lazily-created collections.

use std::cmp::min;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::stream::{Stream, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use qdrant_client::qdrant::UpsertPointsBuilder;
use qdrant_client::serverless::QdrantServerless;
use tokio::time::sleep;
use tokio_stream::wrappers::IntervalStream;

use super::args::ServerlessUploadArgs;
use super::client::create_clients;
use super::collections::CollectionRegistry;
use super::convert::to_serverless_config;
use super::distribution::CollectionPicker;
use crate::args::Args;
use crate::client::retry_with_clients;
use crate::config;
use crate::config::examples::{ExampleKind, resolve};
use crate::dataset;
use crate::generators::{ConfigGenerator, PointGenerator};
use crate::processor::Timing;
use crate::results::{BenchmarkResults, UploadPhase};
use crate::save_jsonl::save_timings_as_jsonl;
use crate::stats::{print_stats, throttler};

/// One upsert request: `count` points of collection slot `coll_idx`, with
/// point ids starting at `start_id` (unique across the whole upload).
#[derive(Debug, Clone, Copy)]
struct Batch {
    coll_idx: usize,
    start_id: u64,
    count: usize,
}

/// Split the per-collection allocation into upsert batches. Point ids are
/// laid out contiguously across collections starting at `offset`, so with a
/// dataset source every collection gets a different slice of the data.
fn plan_batches(per_collection: &[usize], batch_size: usize, offset: usize) -> Vec<Batch> {
    let mut batches = Vec::new();
    let mut next_id = offset as u64;
    for (coll_idx, &count) in per_collection.iter().enumerate() {
        let mut remaining = count;
        while remaining > 0 {
            let n = min(batch_size, remaining);
            batches.push(Batch {
                coll_idx,
                start_id: next_id,
                count: n,
            });
            next_id += n as u64;
            remaining -= n;
        }
    }
    batches
}

/// Request pacing: `--rps` fires at a fixed interval regardless of how many
/// requests are in flight (missed ticks are skipped, not burst); otherwise
/// the regular `--throttle` stream.
fn pacer(args: &Args) -> Box<dyn Stream<Item = ()> + Unpin> {
    match args.rps.filter(|rps| *rps > 0.0 && rps.is_finite()) {
        Some(rps) => {
            let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / rps));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            Box::new(IntervalStream::new(interval).map(|_| ()))
        }
        None => throttler(args.throttle),
    }
}

/// Records the first failure of a run; later ones are only logged.
#[derive(Default)]
struct FirstError(Mutex<Option<anyhow::Error>>);

impl FirstError {
    fn record(&self, err: anyhow::Error) {
        let mut slot = self.0.lock().unwrap();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    fn into_result(self) -> Result<()> {
        match self.0.into_inner().unwrap() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

struct Uploader {
    args: Args,
    clients: Vec<QdrantServerless>,
    registry: CollectionRegistry,
    generator: Box<dyn PointGenerator>,
    bar: ProgressBar,
    started: Instant,
    /// Server-reported upsert latency per batch.
    timings: Mutex<Vec<Timing>>,
    /// Points acknowledged by the server.
    uploaded: Mutex<usize>,
}

impl Uploader {
    async fn upsert(&self, batch_no: usize, batch: Batch) -> Result<()> {
        let client = &self.clients[batch_no % self.clients.len()];
        let name = self.registry.ensure(client, batch.coll_idx).await?;

        let points: Vec<_> = (0..batch.count as u64)
            .map(|i| self.generator.make_point(batch.start_id + i))
            .collect();

        let mut request =
            UpsertPointsBuilder::new(name.clone(), points).wait(self.args.wait_on_upsert);
        if let Some(timeout) = self.args.timeout {
            request = request.timeout(timeout as u64);
        }
        let request = request.build();

        let resp = retry_with_clients(&self.clients, &self.args, |c| {
            c.upsert_points(request.clone())
        })
        .await
        .with_context(|| format!("upsert into {name}"))?;

        self.timings.lock().unwrap().push(Timing {
            delay_millis: self.started.elapsed().as_millis() as u32,
            value: resp.time as f32,
        });
        *self.uploaded.lock().unwrap() += batch.count;
        if resp.time > self.args.timing_threshold {
            self.bar
                .println(format!("Slow upsert on {name}: {:?}", resp.time));
        }
        self.bar.inc(batch.count as u64);

        if let Some(delay_millis) = self.args.delay {
            sleep(Duration::from_millis(delay_millis as u64)).await;
        }
        Ok(())
    }
}

pub async fn run(
    args: &Args,
    upload: ServerlessUploadArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    anyhow::ensure!(
        upload.collections_count > 0,
        "--collections-count must be > 0"
    );

    let resolved = resolve(
        upload.config.file.as_deref(),
        upload.config.example.as_deref(),
        ExampleKind::Upload,
    )?;
    let upload_config = config::parse(&resolved.yaml, &resolved.origin)?;
    let serverless_config = to_serverless_config(&upload_config)?;

    // Dataset-backed configs cap the total at what the dataset holds.
    let total_points = dataset::resolve_num_vectors(
        upload.total_points.or(args.num_vectors),
        args.offset,
        &upload_config,
        &dataset::default_datasets_dir(),
    )?;

    let mut args = args.clone();
    args.num_vectors = Some(total_points);
    args.collection_name = format!("{}*", upload.collection_prefix);
    let mut results = BenchmarkResults::new(&args, Some(resolved.origin));

    let clients = create_clients(&args)?;
    let registry = CollectionRegistry::bootstrap(
        &clients[0],
        &upload.collection_prefix,
        upload.collections_count,
        serverless_config,
    )
    .await?;

    let picker = CollectionPicker::new(upload.collections_count, upload.distribution.into())?;
    let per_collection = picker.allocate(total_points, &mut rand::rng());

    println!(
        "Uploading {total_points} points across {} collections ({:?})",
        upload.collections_count, upload.distribution
    );
    for (i, &n) in per_collection.iter().enumerate().take(5) {
        println!("  {} → {n} points", registry.name(i));
    }
    if per_collection.len() > 5 {
        println!("  …");
    }

    let batches = plan_batches(&per_collection, args.batch_size, args.offset);
    let generator: Box<dyn PointGenerator> = Box::new(ConfigGenerator::new(&upload_config)?);

    let logger = env_logger::Builder::from_default_env().build();
    let multiprogress = MultiProgress::new();
    indicatif_log_bridge::LogWrapper::new(multiprogress.clone(), logger)
        .try_init()
        .ok();

    let bar = multiprogress.add(ProgressBar::new(total_points as u64));
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("progress style"),
    );
    bar.set_draw_target(ProgressDrawTarget::stdout_with_hz(2));

    // `--rps` decides the send rate on its own, so concurrency is unbounded;
    // otherwise `-p` bounds the number of in-flight batches.
    let concurrency = if args.rps.is_some() {
        None
    } else {
        Some(args.parallel.max(1))
    };
    let pacer = pacer(&args);
    let start_timestamp_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64;

    let uploader = Uploader {
        args: args.clone(),
        clients,
        registry,
        generator,
        bar: bar.clone(),
        started: Instant::now(),
        timings: Mutex::new(Vec::new()),
        uploaded: Mutex::new(0),
    };
    let first_error = FirstError::default();

    futures::stream::iter(batches.into_iter().enumerate())
        .take_while(|_| futures::future::ready(!stopped.load(Ordering::Relaxed)))
        .zip(pacer)
        .for_each_concurrent(concurrency, |((batch_no, batch), _)| {
            let uploader = &uploader;
            let first_error = &first_error;
            let stopped = &stopped;
            let args = &args;
            async move {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }
                if let Err(err) = uploader.upsert(batch_no, batch).await {
                    uploader.bar.println(format!("Error: {err:?}"));
                    if !args.ignore_errors {
                        first_error.record(err);
                        stopped.store(true, Ordering::Relaxed);
                    }
                }
            }
        })
        .await;

    let duration_secs = uploader.started.elapsed().as_secs_f64();
    if stopped.load(Ordering::Relaxed) {
        bar.abandon();
    } else {
        bar.finish();
    }

    let uploaded = *uploader.uploaded.lock().unwrap();
    let phase = UploadPhase::new(duration_secs, uploaded);
    println!(
        "Uploaded {} points in {:.3} s ({:.0} points/s)",
        phase.num_points, phase.duration_secs, phase.points_per_sec
    );
    uploader.registry.summary();

    let mut timings = uploader.timings.into_inner().unwrap();
    println!("--- Upsert timings ---");
    print_stats(&args, &mut timings, "upsert time", true);
    if let Some(jsonl_path) = &args.jsonl_updates {
        save_timings_as_jsonl(
            jsonl_path,
            args.absolute_time.unwrap_or(false),
            &timings,
            start_timestamp_millis,
            "upsert_latency",
        )?;
    }

    first_error.into_result()?;
    results.results.upload = Some(phase);
    results.write_if_requested(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_cover_every_point_with_unique_ids() {
        let batches = plan_batches(&[5, 0, 7], 3, 100);
        let counts: Vec<_> = batches.iter().map(|b| (b.coll_idx, b.count)).collect();
        assert_eq!(counts, vec![(0, 3), (0, 2), (2, 3), (2, 3), (2, 1)]);
        let ids: Vec<_> = batches.iter().map(|b| b.start_id).collect();
        assert_eq!(ids, vec![100, 103, 105, 108, 111]);
    }
}
