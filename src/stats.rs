use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};
use tokio::join;

use crate::args::Args;
use crate::common::{Timing, throttler};
use crate::processor::Processor;
use crate::save_jsonl::save_timings_as_jsonl;

#[derive(Serialize, Deserialize)]
pub struct SearcherResults {
    pub server_timings: Vec<f64>,
    pub rps: Vec<f64>,
    pub full_timings: Vec<f64>,
}

pub fn write_to_json(path: &String, results: SearcherResults) {
    let mut file = File::create(path).unwrap();
    serde_json::to_writer(&mut file, &results).unwrap();
    println!("Search results written to {path}");
}

pub fn print_stats(args: &Args, values: &mut [Timing], metric_name: &str, show_percentiles: bool) {
    if values.is_empty() {
        return;
    }
    // sort values in ascending order
    values.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

    let avg_time: f64 = values.iter().map(|x| x.value).sum::<f64>() / values.len() as f64;
    let min_time: f64 = values.first().unwrap().value;
    let max_time: f64 = values.last().unwrap().value;
    let p50_time: f64 = values[(values.len() as f32 * 0.50) as usize].value;

    println!("Min {metric_name}: {min_time}");
    println!("Avg {metric_name}: {avg_time}");
    println!("Median {metric_name}: {p50_time}");

    if show_percentiles {
        let p95_time: f64 = values[(values.len() as f32 * 0.95) as usize].value;
        println!("p95 {metric_name}: {p95_time}");

        for digits in 2..=args.p9 {
            let factor = 1.0 - 1.0 * 0.1f64.powf(digits as f64);
            let index = ((values.len() as f64 * factor) as usize).min(values.len() - 1);
            let nines = "9".repeat(digits);
            let time = values[index].value;
            println!("p{nines} {metric_name}: {time}");
        }
    }

    println!("Max {metric_name}: {max_time}");
}

pub async fn process<P: Processor + Sync>(
    args: &Args,
    stopped: Arc<AtomicBool>,
    processor: P,
) -> Result<()> {
    let batch_size = processor.get_batch_size();
    let batch_count = args.num_vectors.div_ceil(batch_size);

    let multiprogress = MultiProgress::new();
    let progress_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));
    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    progress_bar.set_style(progress_style);
    // Refresh bar 2 times per seconds
    let draw_target = ProgressDrawTarget::stdout_with_hz(2);
    progress_bar.set_draw_target(draw_target);

    // Use RPS mode if --rps is set, otherwise use parallel mode
    if let Some(target_rps) = args.rps {
        process_with_rps(
            args,
            stopped.clone(),
            &processor,
            &progress_bar,
            batch_count,
            batch_size,
            target_rps,
        )
        .await?;
    } else {
        process_with_parallel(
            args,
            stopped.clone(),
            &processor,
            &progress_bar,
            batch_count,
            batch_size,
        )
        .await?;
    }

    if stopped.load(Ordering::Relaxed) {
        progress_bar.abandon();
    } else {
        progress_bar.finish();
    }

    let mut full_timings = processor.full_timings();
    println!("--- Request timings ---");
    print_stats(args, &mut full_timings, "request time", true);
    let mut server_timings = processor.server_timings();
    println!("--- Server timings ---");
    print_stats(args, &mut server_timings, "server time", true);

    let mut rps = processor.rps();
    println!("--- RPS ---");
    print_stats(args, &mut rps, "rps", false);
    let mut qps = processor.qps();
    println!("--- QPS ---");
    print_stats(args, &mut qps, "qps", false);

    let precisions = processor.precisions();
    if !precisions.is_empty() {
        println!("--- Precision ---");
        let avg_precision = precisions.iter().sum::<f32>() / precisions.len() as f32;
        println!("Avg precision@10: {avg_precision}");

        let mut sorted_precisions = precisions;
        sorted_precisions.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_time = sorted_precisions[(sorted_precisions.len() as f32 * 0.50) as usize];
        println!("Median precision@10: {p50_time}");
    }

    if let Some(json) = args.json.as_ref() {
        println!("--- Writing results to json file ---");
        write_to_json(
            json,
            SearcherResults {
                server_timings: server_timings.iter().map(|x| x.value).collect(),
                rps: rps.iter().map(|x| x.value).collect(),
                full_timings: full_timings.iter().map(|x| x.value).collect(),
            },
        );
    }

    if let Some(jsonl_path) = &args.jsonl_searches {
        save_timings_as_jsonl(
            jsonl_path,
            args.absolute_time.unwrap_or(false),
            &server_timings,
            processor.start_timestamp_millis(),
            "request_latency",
        )?;
    }

    if let Some(jsonl_path) = &args.jsonl_rps {
        save_timings_as_jsonl(
            jsonl_path,
            args.absolute_time.unwrap_or(false),
            &rps,
            processor.start_timestamp_millis(),
            "request_rps",
        )?;
    }

    Ok(())
}

/// Process requests using fixed parallelism (original behavior)
async fn process_with_parallel<P: Processor>(
    args: &Args,
    stopped: Arc<AtomicBool>,
    processor: &P,
    progress_bar: &ProgressBar,
    batch_count: usize,
    batch_size: usize,
) -> Result<()> {
    let query_stream = (0..batch_count)
        .take_while(|_| !stopped.load(Ordering::Relaxed))
        .map(|n| {
            let future = processor.make_request(n, args, progress_bar);
            progress_bar.inc(batch_size as u64);
            future
        });

    let mut throttler = throttler(args.throttle);
    let mut search_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let (Some(()), Some(result)) = { join!(throttler.next(), search_stream.next()) } {
        // Continue with no error
        let err = match result {
            Ok(()) => continue,
            Err(err) => err,
        };

        if args.ignore_errors {
            progress_bar.println(format!("Error: {err}"));
        } else {
            return Err(err);
        }
    }
    Ok(())
}

/// Process requests at a fixed rate (RPS mode)
/// Spawns requests at regular intervals regardless of how many are in-flight.
async fn process_with_rps<P: Processor + Sync>(
    args: &Args,
    stopped: Arc<AtomicBool>,
    processor: &P,
    progress_bar: &ProgressBar,
    batch_count: usize,
    batch_size: usize,
    target_rps: f64,
) -> Result<()> {
    use futures::stream::FuturesUnordered;

    let interval_duration = Duration::from_secs_f64(1.0 / target_rps);
    let mut interval = tokio::time::interval(interval_duration);
    // Don't burst if we fall behind - skip missed ticks
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    let mut requests_sent = 0usize;
    let mut first_error: Option<anyhow::Error> = None;

    loop {
        if stopped.load(Ordering::Relaxed) {
            break;
        }

        // Check if we've sent all requests
        let all_sent = requests_sent >= batch_count;

        if all_sent && in_flight.is_empty() {
            // All done
            break;
        }

        tokio::select! {
            // Wait for next interval tick to send a new request
            _ = interval.tick(), if !all_sent => {
                if stopped.load(Ordering::Relaxed) {
                    break;
                }

                let req_id = requests_sent;
                requests_sent += 1;
                progress_bar.inc(batch_size as u64);

                let future = processor.make_request(req_id, args, progress_bar);
                in_flight.push(future);
            }
            // Process completed requests
            Some(Err(err)) = in_flight.next(), if !in_flight.is_empty() => {
                if args.ignore_errors {
                    progress_bar.println(format!("Error: {err}"));
                } else if first_error.is_none() {
                    first_error = Some(err);
                    // Stop sending new requests on error
                    break;
                }
            }
            else => {
                // If both branches are disabled, we need to wait for in_flight
                if let Some(Err(err)) = in_flight.next().await {
                    if args.ignore_errors {
                        progress_bar.println(format!("Error: {err}"));
                    } else if first_error.is_none() {
                        first_error = Some(err);
                        break;
                    }
                }
            }
        }
    }

    // Drain remaining in-flight requests
    while let Some(Err(err)) = in_flight.next().await {
        if args.ignore_errors {
            progress_bar.println(format!("Error: {err}"));
        } else if first_error.is_none() {
            first_error = Some(err);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(())
}
