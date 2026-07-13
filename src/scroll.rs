use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScrollPointsBuilder;

use rand::{Rng, RngExt};

use crate::args::Args;
use crate::client::retry_with_clients;
use crate::config::scroll::ScrollConfig;
use crate::generators::queries::FilterGenerator;
use crate::generators::random::{DEFAULT_VOCAB_SIZE, create_zipf, random_filter};
use crate::processor::{Processor, Timing};

#[derive(Debug, Default)]
struct ScrollStats {
    server_timings: Vec<Timing>,
    rps: Vec<Timing>,
    full_timings: Vec<Timing>,
}

/// Where a scroll request's filter comes from.
enum Filters {
    /// Legacy `bfb --scroll`: filters come from the payload flags.
    Flags {
        uuids: Vec<String>,
        zipf: Option<rand_distr::Zipf<f64>>,
    },
    /// `bfb scroll --file`: one generator per `requests:` template, one picked
    /// at random per request.
    Config(Vec<FilterGenerator>),
}

pub struct ScrollProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    stats: Mutex<ScrollStats>,
    filters: Filters,
}

impl ScrollProcessor {
    /// Flag-driven: filters come from `--keywords`, `--geo-payloads`, ….
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        uuids: Vec<String>,
    ) -> Self {
        let zipf = args
            .text_payloads
            .then(|| create_zipf(args.text_payload_vocabulary.unwrap_or(DEFAULT_VOCAB_SIZE)));

        Self::with_filters(args, stopped, clients, Filters::Flags { uuids, zipf })
    }

    /// Config-driven: filters come from the YAML `requests:` templates.
    pub fn from_config(
        args: Args,
        config: &ScrollConfig,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
    ) -> Self {
        let mut rng = rand::rng();
        let generators = config
            .requests
            .iter()
            .map(|request| FilterGenerator::new(&request.filters, &mut rng))
            .collect();

        Self::with_filters(args, stopped, clients, Filters::Config(generators))
    }

    fn with_filters(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        filters: Filters,
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
            stats: Mutex::new(ScrollStats::default()),
            filters,
        }
    }

    fn build_filter(
        &self,
        rng: &mut impl Rng,
        args: &Args,
    ) -> Option<qdrant_client::qdrant::Filter> {
        match &self.filters {
            Filters::Flags { uuids, zipf } => random_filter(
                rng,
                &self.args.keywords,
                &self.args.float_payloads,
                &self.args.int_payloads,
                uuids,
                self.args.match_any,
                self.args.geo_payloads,
                self.args.bool_payloads,
                args.keywords_length_multiplier,
                zipf.as_ref(),
            ),
            Filters::Config(generators) if !generators.is_empty() => {
                generators[rng.random_range(0..generators.len())].build(rng)
            }
            // `ScrollConfig::validate` rejects an empty `requests:` list, so this
            // is unreachable today; scroll unfiltered rather than panic if a
            // future caller builds a processor without going through it.
            Filters::Config(_) => None,
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
        let mut rng = rand::rng();
        let query_filter = self.build_filter(&mut rng, args);

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

        if let Some(timeout) = self.args.timeout {
            request_builder = request_builder.timeout(timeout as u64);
        }

        let request = request_builder.build();
        let res = retry_with_clients(&self.clients, args, |client| client.scroll(request.clone()))
            .await?;

        let elapsed = start.elapsed().as_secs_f32();
        let delay_millis = self.start_time.elapsed().as_millis() as u32;
        let full_timing = Timing {
            delay_millis,
            value: elapsed,
        };

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
            delay_millis,
            value: res.time as f32,
        };

        let rps_timing = Timing {
            delay_millis,
            value: progress_bar.per_sec() as f32,
        };

        {
            let mut stats = self.stats.lock().unwrap();
            stats.full_timings.push(full_timing);
            stats.server_timings.push(server_timing);
            stats.rps.push(rps_timing);
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
        self.stats.lock().unwrap().rps.clone()
    }

    fn full_timings(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().full_timings.clone()
    }

    fn get_batch_size(&self) -> usize {
        1 // No batching for scroll.
    }
}
