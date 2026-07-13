use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use tokio::join;

use futures::Stream;
use tokio_stream::wrappers::IntervalStream;

use crate::args::Args;
use crate::processor::{Processor, Timing};
use crate::results::{PrecisionSummary, QueryPhase, Summary};
use crate::save_jsonl::save_timings_as_jsonl;

/// Build a stream that will emit a unit value at the given frequency
///
/// If `None` - the stream will emit a unit value every time it is polled.
pub(crate) fn throttler(hz: Option<f32>) -> Box<dyn Stream<Item = ()> + Unpin> {
    match hz
        // Do not support zero or infinite
        .filter(|throttle| *throttle != 0.0 && !throttle.is_nan() && !throttle.is_infinite())
        .map(|throttle| Duration::from_secs_f32(1.0 / throttle))
        // Do not support durations of zero
        .filter(|duration| !duration.is_zero())
    {
        Some(duration) => {
            let interval = tokio::time::interval(duration);
            Box::new(IntervalStream::new(interval).map(|_| ()))
        }
        None => Box::new(futures::stream::repeat(())),
    }
}

/// Sort `values`, print their distribution, and return it as a [`Summary`].
/// Returns `None` for an empty series.
pub fn print_stats(
    args: &Args,
    values: &mut [Timing],
    metric_name: &str,
    show_percentiles: bool,
) -> Option<Summary> {
    if values.is_empty() {
        return None;
    }
    // sort values in ascending order
    values.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

    let summary = Summary::from_sorted(values).expect("non-empty");

    println!("Min {metric_name}: {}", summary.min);
    println!("Avg {metric_name}: {}", summary.avg);
    println!("Median {metric_name}: {}", summary.p50);

    if show_percentiles {
        println!("p95 {metric_name}: {}", summary.p95);

        for digits in 2..=args.p9 {
            let factor = 1.0 - 0.1f64.powf(digits as f64);
            let index = ((values.len() as f64 * factor) as usize).min(values.len() - 1);
            let nines = "9".repeat(digits);
            let time = values[index].value;
            println!("p{nines} {metric_name}: {time}");
        }
    }

    println!("Max {metric_name}: {}", summary.max);
    Some(summary)
}

/// Run `processor` to completion and return the phase's measurements.
pub async fn process<P: Processor + Sync>(
    args: &Args,
    stopped: Arc<AtomicBool>,
    processor: P,
) -> Result<QueryPhase> {
    let started = Instant::now();
    let batch_size = processor.get_batch_size();
    let batch_count = args.num_vectors_or_default().div_ceil(batch_size);

    let multiprogress = MultiProgress::new();
    let progress_bar = multiprogress.add(ProgressBar::new(args.num_vectors_or_default() as u64));
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
    let duration_secs = started.elapsed().as_secs_f64();

    let mut full_timings = processor.full_timings();
    println!("--- Request timings ---");
    let request_time = print_stats(args, &mut full_timings, "request time", true);
    let mut server_timings = processor.server_timings();
    println!("--- Server timings ---");
    let server_time = print_stats(args, &mut server_timings, "server time", true);

    let mut rps = processor.rps();
    println!("--- RPS ---");
    print_stats(args, &mut rps, "rps", false);
    let mut qps = processor.qps();
    println!("--- QPS ---");
    print_stats(args, &mut qps, "qps", false);

    let precision = PrecisionSummary::new(processor.precisions());
    if let Some(precision) = &precision {
        println!("--- Precision ---");
        println!("Avg precision@10: {}", precision.avg);
        println!("Median precision@10: {}", precision.p50);
    }

    let phase = QueryPhase {
        duration_secs,
        server_timings: server_timings.iter().map(|x| x.value).collect(),
        full_timings: full_timings.iter().map(|x| x.value).collect(),
        rps: rps.iter().map(|x| x.value).collect(),
        qps: qps.iter().map(|x| x.value).collect(),
        server_time,
        request_time,
        precision,
    };

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

    Ok(phase)
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn timings(values: &[f32]) -> Vec<Timing> {
        values
            .iter()
            .map(|&value| Timing {
                delay_millis: 0,
                value,
            })
            .collect()
    }

    #[test]
    fn print_stats_summarizes_and_sorts_in_place() {
        let args = Args::parse_from(["bfb"]);
        let mut values = timings(&[3.0, 1.0, 2.0, 4.0]);

        let summary = print_stats(&args, &mut values, "request time", true).unwrap();

        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 4.0);
        assert_eq!(summary.avg, 2.5);
        // The caller relies on `print_stats` leaving the series sorted.
        let sorted: Vec<_> = values.iter().map(|t| t.value).collect();
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn print_stats_of_empty_series_is_none() {
        let args = Args::parse_from(["bfb"]);
        assert!(print_stats(&args, &mut [], "request time", true).is_none());
    }
}
