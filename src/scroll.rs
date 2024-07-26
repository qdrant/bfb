use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::Qdrant;

use crate::common::{retry_with_clients, Timing};
use crate::processor::Processor;
use crate::{random_filter, Args};

pub struct ScrollProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    pub server_timings: Mutex<Vec<Timing>>,
    pub rps: Mutex<Vec<Timing>>,
    pub full_timings: Mutex<Vec<Timing>>,
}

impl ScrollProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, clients: Vec<Qdrant>) -> Self {
        ScrollProcessor {
            args,
            stopped,
            clients,
            start_timestamp_millis: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as f64,
            start_time: std::time::Instant::now(),
            server_timings: Mutex::new(Vec::new()),
            rps: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
        }
    }

    pub async fn scroll(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let start = std::time::Instant::now();

        let query_filter = random_filter(
            self.args.keywords.first().cloned(),
            self.args.float_payloads.first().cloned().unwrap_or(false),
            self.args.int_payloads.first().cloned(),
            self.args.match_any,
        );

        let mut request_builder = ScrollPointsBuilder::new(self.args.collection_name.clone())
            .limit(self.args.search_limit as u32)
            .with_payload(self.args.search_with_payload);

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

        self.full_timings.lock().unwrap().push(full_timing);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow scroll: {:?}", res.time));
        }

        if res.result.len() < self.args.search_limit {
            progress_bar.println(format!(
                "Scroll result is too small: {} of {}",
                res.result.len(),
                self.args.search_limit
            ));
        }

        let server_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: res.time,
        };

        let rps_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: progress_bar.per_sec(),
        };

        self.server_timings.lock().unwrap().push(server_timing);
        self.rps.lock().unwrap().push(rps_timing);

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
        self.server_timings.lock().unwrap().clone()
    }

    fn rps(&self) -> Vec<Timing> {
        self.rps.lock().unwrap().clone()
    }

    fn full_timings(&self) -> Vec<Timing> {
        self.full_timings.lock().unwrap().clone()
    }
}
