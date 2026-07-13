use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use tokio::join;

use crate::args::Args;
use crate::client::get_config;
use crate::config::UploadConfig;
use crate::fbin_reader::FBinReader;
use crate::generators::{ConfigGenerator, LegacyGenerator, PointGenerator};
use crate::results::UploadPhase;
use crate::stats::throttler;
use crate::upsert::UpsertProcessor;

/// Legacy flag-driven upload.
pub async fn upload_data(args: &Args, stopped: Arc<AtomicBool>) -> Result<UploadPhase> {
    let reader = args
        .fbin
        .as_ref()
        .map(|path| FBinReader::new(Path::new(path)));
    let generator = Arc::new(LegacyGenerator::new(args.clone(), reader));
    run_upload(args, stopped, generator).await
}

/// YAML-config-driven upload (`bfb upload --file config.yaml`).
pub async fn upload_with_config(
    args: &Args,
    config: &UploadConfig,
    stopped: Arc<AtomicBool>,
) -> Result<UploadPhase> {
    let generator = Arc::new(ConfigGenerator::new(config)?);
    run_upload(args, stopped, generator).await
}

async fn run_upload(
    args: &Args,
    stopped: Arc<AtomicBool>,
    generator: Arc<dyn PointGenerator>,
) -> Result<UploadPhase> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(qdrant_client::Qdrant::new(config)?);
    }

    let logger = env_logger::Builder::from_default_env().build();

    let multiprogress = MultiProgress::new();

    indicatif_log_bridge::LogWrapper::new(multiprogress.clone(), logger)
        .try_init()
        .unwrap();

    let sent_bar = multiprogress.add(ProgressBar::new(args.num_vectors_or_default() as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    sent_bar.set_style(progress_style);
    // Refresh bar 2 times per seconds
    let draw_target = ProgressDrawTarget::stdout_with_hz(2);
    sent_bar.set_draw_target(draw_target);

    let sent_bar_arc = Arc::new(sent_bar);

    let upserter = UpsertProcessor::new(
        args.clone(),
        stopped.clone(),
        clients,
        sent_bar_arc.clone(),
        generator,
    );

    let num_vectors = args.num_vectors_or_default();
    let num_batches = num_vectors.div_ceil(args.batch_size);

    if stopped.load(Ordering::Relaxed) {
        sent_bar_arc.abandon();
        return Ok(UploadPhase::new(0.0, 0));
    }

    // Time only the upload itself: client setup and generator construction
    // (which may download a dataset) are not part of upload throughput.
    let started = Instant::now();

    // Use RPS mode if --rps is set, otherwise use parallel mode
    if let Some(target_rps) = args.rps {
        upload_with_rps(
            args,
            stopped.clone(),
            &upserter,
            &sent_bar_arc,
            num_batches,
            target_rps,
        )
        .await?;
    } else {
        upload_with_parallel(args, stopped.clone(), &upserter, &sent_bar_arc, num_batches).await?;
    }

    let duration_secs = started.elapsed().as_secs_f64();

    if stopped.load(Ordering::Relaxed) {
        sent_bar_arc.abandon();
    } else {
        sent_bar_arc.finish();
    }

    upserter.save_data();

    // The bar advances a whole batch at a time, so the final batch can push it
    // past the requested count; report what was actually asked for.
    let num_points = (sent_bar_arc.position() as usize).min(num_vectors);
    let phase = UploadPhase::new(duration_secs, num_points);
    println!(
        "Uploaded {} points in {:.3} s ({:.0} points/s)",
        phase.num_points, phase.duration_secs, phase.points_per_sec
    );

    Ok(phase)
}

/// Upload data using fixed parallelism (original behavior)
async fn upload_with_parallel(
    args: &Args,
    stopped: Arc<AtomicBool>,
    upserter: &UpsertProcessor,
    sent_bar_arc: &Arc<ProgressBar>,
    num_batches: usize,
) -> Result<()> {
    let query_stream = (0..num_batches)
        .take_while(|_| !stopped.load(Ordering::Relaxed))
        .map(|n| {
            let future = upserter.upsert(n, args);
            sent_bar_arc.inc(args.batch_size as u64);
            future
        });

    let mut throttler = throttler(args.throttle);
    let mut upsert_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let (Some(()), Some(result)) = { join!(throttler.next(), upsert_stream.next()) } {
        result?;
    }
    Ok(())
}

/// Upload data at a fixed rate (RPS mode)
async fn upload_with_rps(
    args: &Args,
    stopped: Arc<AtomicBool>,
    upserter: &UpsertProcessor,
    sent_bar_arc: &Arc<ProgressBar>,
    num_batches: usize,
    target_rps: f64,
) -> Result<()> {
    use futures::stream::FuturesUnordered;

    let interval_duration = Duration::from_secs_f64(1.0 / target_rps);
    let mut interval = tokio::time::interval(interval_duration);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    let mut batches_sent = 0usize;
    let mut first_error: Option<anyhow::Error> = None;

    loop {
        if stopped.load(Ordering::Relaxed) {
            break;
        }

        let all_sent = batches_sent >= num_batches;

        if all_sent && in_flight.is_empty() {
            break;
        }

        tokio::select! {
            _ = interval.tick(), if !all_sent => {
                if stopped.load(Ordering::Relaxed) {
                    break;
                }

                let batch_id = batches_sent;
                batches_sent += 1;
                sent_bar_arc.inc(args.batch_size as u64);

                let future = upserter.upsert(batch_id, args);
                in_flight.push(future);
            }
            Some(Err(err)) = in_flight.next(), if !in_flight.is_empty() => {
                if first_error.is_none() {
                    first_error = Some(err);
                    break;
                }
            }
            else => {
                if let Some(Err(err)) = in_flight.next().await
                    && first_error.is_none()
                {
                    first_error = Some(err);
                    break;
                }
            }
        }
    }

    // Drain remaining in-flight requests
    while let Some(Err(err)) = in_flight.next().await {
        if first_error.is_none() {
            first_error = Some(err);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(())
}
