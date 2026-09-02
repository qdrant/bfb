//! `bfb serverless upload` — spread points across lazily-created collections.

use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use qdrant_client::qdrant::UpsertPointsBuilder;
use tokio::time::sleep;

use super::args::ServerlessUploadArgs;
use super::client::{self, retry_with_clients};
use super::collections::CollectionRegistry;
use super::convert::to_serverless_config;
use super::distribution::CollectionPicker;
use crate::args::Args;
use crate::config;
use crate::generators::{ConfigGenerator, PointGenerator};
use crate::stats::throttler;

pub async fn run(
    args: &Args,
    upload: ServerlessUploadArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    anyhow::ensure!(
        upload.collections_count > 0,
        "--collections-count must be > 0"
    );

    let yaml = std::fs::read_to_string(&upload.config_file)
        .with_context(|| format!("read config {}", upload.config_file))?;
    let upload_config = config::parse(&yaml, &upload.config_file)?;
    let serverless_config = to_serverless_config(&upload_config)?;

    let total_points = upload.total_points.or(args.num_vectors).unwrap_or(100_000);

    let clients = client::create_clients(args)?;
    let registry = Arc::new(
        CollectionRegistry::bootstrap(
            &clients[0],
            &upload.collection_prefix,
            upload.collections_count,
            serverless_config,
        )
        .await?,
    );

    let picker = CollectionPicker::new(upload.collections_count, upload.distribution.into())?;
    let mut rng = rand::rng();
    let per_collection = picker.allocate(total_points, &mut rng);

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

    let generator: Arc<dyn PointGenerator> = Arc::new(ConfigGenerator::new(&upload_config)?);

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
    let bar = Arc::new(bar);

    // Flatten (collection_idx, local_point_id) into a work list of batches.
    // Each batch stays within one collection so upserts stay simple.
    let mut batches: Vec<(usize, u64, usize)> = Vec::new(); // (coll_idx, start_id, count)
    for (coll_idx, &count) in per_collection.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let mut remaining = count;
        let mut start = 0u64;
        while remaining > 0 {
            let n = min(args.batch_size, remaining);
            batches.push((coll_idx, start, n));
            start += n as u64;
            remaining -= n;
        }
    }

    let started = Instant::now();
    let parallel = if args.rps.is_some() {
        // RPS mode: one in-flight stream controlled by the throttler below.
        1
    } else {
        args.parallel.max(1)
    };

    let throttler = throttler(args.rps.map(|r| r as f32).or(args.throttle));
    let stopped_flag = stopped.clone();
    let clients = Arc::new(clients);
    let args = args.clone();

    futures::stream::iter(batches.into_iter().enumerate())
        .take_while(|_| {
            let s = stopped_flag.clone();
            async move { !s.load(Ordering::Relaxed) }
        })
        .zip(throttler)
        .for_each_concurrent(parallel, |((batch_no, (coll_idx, start_id, count)), _)| {
            let clients = clients.clone();
            let registry = registry.clone();
            let generator = generator.clone();
            let bar = bar.clone();
            let args = args.clone();
            let stopped = stopped_flag.clone();

            async move {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }

                let client = &clients[batch_no % clients.len()];
                let name = match registry.ensure(client, coll_idx).await {
                    Ok(n) => n,
                    Err(e) => {
                        bar.println(format!("ensure collection failed: {e:?}"));
                        if !args.ignore_errors {
                            stopped.store(true, Ordering::Relaxed);
                        }
                        return;
                    }
                };

                let mut points = Vec::with_capacity(count);
                for i in 0..count {
                    points.push(generator.make_point(start_id + i as u64));
                }

                let mut request =
                    UpsertPointsBuilder::new(name.clone(), points).wait(args.wait_on_upsert);
                if let Some(timeout) = args.timeout {
                    request = request.timeout(timeout as u64);
                }
                let request = request.build();

                let res =
                    retry_with_clients(&clients, &args, |c| c.upsert_points(request.clone())).await;

                match res {
                    Ok(resp) => {
                        if resp.time > args.timing_threshold {
                            bar.println(format!("Slow upsert on {name}: {:?}", resp.time));
                        }
                        registry.mark_queryable(&name);
                        bar.inc(count as u64);
                    }
                    Err(e) => {
                        bar.println(format!("upsert failed on {name}: {e:?}"));
                        if !args.ignore_errors {
                            stopped.store(true, Ordering::Relaxed);
                        }
                    }
                }

                if let Some(delay_millis) = args.delay {
                    sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
                }
            }
        })
        .await;

    bar.finish_and_clear();
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "Serverless upload finished in {elapsed:.2}s ({:.0} points/s)",
        total_points as f64 / elapsed.max(1e-9)
    );
    registry.summary();
    Ok(())
}
