use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScrollPointsBuilder;

use crate::args::Args;
use crate::common::{DEFAULT_VOCAB_SIZE, Timing, random_filter, retry_with_clients};
use crate::processor::Processor;

pub struct ScrollProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    start_timestamp_millis: f64,
    start_time: std::time::Instant,
    uuids: Vec<String>,
    stats: Mutex<ScrollStats>,
}

#[derive(Default)]
pub struct ScrollStats {
    server_timings: Vec<Timing>,
    rps: Vec<Timing>,
    full_timings: Vec<Timing>,
    slow_scroll: Vec<Timing>,
}

impl ScrollProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        uuids: Vec<String>,
    ) -> Self {
        ScrollProcessor {
            args,
            stopped,
            clients,
            start_timestamp_millis: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as f64,
            start_time: std::time::Instant::now(),
            uuids,
            stats: Mutex::new(ScrollStats::default()),
        }
    }

    pub async fn scroll(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let start = std::time::Instant::now();

        let query_filter = random_filter(
            &self.args.keywords,
            &self.args.float_payloads,
            &self.args.int_payloads,
            &self.uuids,
            self.args.match_any,
            self.args.geo_payloads,
            self.args.bool_payloads,
            self.args.text_payloads.then(|| {
                self.args
                    .text_payload_vocabulary
                    .unwrap_or(DEFAULT_VOCAB_SIZE)
            }),
        );

        let mut request_builder = ScrollPointsBuilder::new(self.args.collection_name.clone())
            .limit(self.args.search_limit as u32)
            .with_payload(self.args.search_with_payload)
            .with_vectors(self.args.search_with_vectors);

        if let Some(filter) = query_filter {
            request_builder = request_builder.filter(filter);
        }

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        let request = request_builder.build();
        let res = retry_with_clients(&self.clients, args, |client| client.scroll(request.clone()))
            .await?;

        let elapsed = start.elapsed().as_secs_f64();

        let full_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: elapsed,
        };

        let server_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: res.time,
        };

        let slow_request = res.time > self.args.timing_threshold;
        if slow_request {
            progress_bar.println(format!("Slow scroll: {:?}", res.time));
        }

        if res.result.len() < self.args.search_limit {
            progress_bar.println(format!(
                "Scroll result is too small: {} of {}",
                res.result.len(),
                self.args.search_limit
            ));
        }

        let rps_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: progress_bar.per_sec(),
        };

        // Update stats all at once
        {
            let mut stats = self.stats.lock().unwrap();
            stats.full_timings.push(full_timing);
            stats.server_timings.push(server_timing);
            stats.rps.push(rps_timing);

            if slow_request {
                stats.slow_scroll.push(server_timing);
            }
        }

        if let Some(delay_millis) = self.args.delay {
            tokio::time::sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        Ok(())
    }
}

impl Processor for ScrollProcessor {
    async fn make_request(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        self.scroll(_req_id, args, progress_bar).await
    }

    fn start_timestamp_millis(&self) -> f64 {
        self.start_timestamp_millis
    }

    fn server_timings(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().server_timings.clone()
    }

    fn qps(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().rps.clone()
    }

    fn rps(&self) -> Vec<Timing> {
        // for requests without batching, qps = rps
        self.stats.lock().unwrap().rps.clone()
    }

    fn full_timings(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().full_timings.clone()
    }

    fn get_batch_size(&self) -> usize {
        1 // No batching for scroll.
    }

    fn slow_requests(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().slow_scroll.clone()
    }
}
