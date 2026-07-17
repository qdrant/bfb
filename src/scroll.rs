use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::{PointId, Query, QueryPointsBuilder, Sample, ScrollPointsBuilder};

use rand::{Rng, RngExt};

use crate::args::Args;
use crate::client::retry_with_clients;
use crate::config::scroll::{ScrollConfig, ScrollMode};
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

/// A `Sequential` walk in progress. The filter is held for the whole walk: a
/// cursor only means anything against the query that produced it.
struct Walk {
    filter: Option<qdrant_client::qdrant::Filter>,
    offset: PointId,
}

pub struct ScrollProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    stats: Mutex<ScrollStats>,
    filters: Filters,
    mode: ScrollMode,
    /// `Sequential` only: a pool of walks in progress. A request checks one out for
    /// the duration of its scroll and returns it advanced, so a walk is never shared.
    /// The pool sizes itself: with N requests in flight there are at most N walks.
    walks: Mutex<Vec<Walk>>,
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

        Self::with_filters(
            args,
            stopped,
            clients,
            Filters::Flags { uuids, zipf },
            ScrollMode::default(),
        )
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

        Self::with_filters(
            args,
            stopped,
            clients,
            Filters::Config(generators),
            config.mode,
        )
    }

    fn with_filters(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        filters: Filters,
        mode: ScrollMode,
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
            mode,
            walks: Mutex::new(Vec::new()),
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

    /// One scroll request. Returns `(points, server_secs, next_page_offset)`.
    async fn scroll_cursor(
        &self,
        args: &Args,
        filter: Option<qdrant_client::qdrant::Filter>,
        offset: Option<PointId>,
    ) -> Result<(usize, f64, Option<PointId>), anyhow::Error> {
        let mut request_builder = ScrollPointsBuilder::new(self.args.collection_name.clone())
            .limit(self.args.search_limit as u32)
            .with_payload(self.args.search_with_payload)
            .with_vectors(self.args.search_with_vectors);

        if let Some(filter) = filter {
            request_builder = request_builder.filter(filter);
        }

        if let Some(offset) = offset {
            request_builder = request_builder.offset(offset);
        }

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        if let Some(shard_key) = &self.args.shard_key {
            request_builder =
                request_builder.shard_key_selector(vec![Key::Keyword(shard_key.clone())]);
        }

        if let Some(timeout) = self.args.timeout {
            request_builder = request_builder.timeout(timeout as u64);
        }

        let request = request_builder.build();
        let res = retry_with_clients(&self.clients, args, |client| client.scroll(request.clone()))
            .await?;

        Ok((res.result.len(), res.time, res.next_page_offset))
    }

    /// A vector-less `query` with `sample: random` — a randomly-sampled page.
    async fn sample(
        &self,
        args: &Args,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<(usize, f64), anyhow::Error> {
        let mut request_builder = QueryPointsBuilder::new(self.args.collection_name.clone())
            .query(Query::new_sample(Sample::Random))
            .limit(self.args.search_limit as u64)
            .with_payload(self.args.search_with_payload)
            .with_vectors(self.args.search_with_vectors);

        if let Some(filter) = filter {
            request_builder = request_builder.filter(filter);
        }

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        if let Some(shard_key) = &self.args.shard_key {
            request_builder =
                request_builder.shard_key_selector(vec![Key::Keyword(shard_key.clone())]);
        }

        if let Some(timeout) = self.args.timeout {
            request_builder = request_builder.timeout(timeout as u64);
        }

        let request = request_builder.build();
        let res =
            retry_with_clients(&self.clients, args, |client| client.query(request.clone())).await?;

        Ok((res.result.len(), res.time))
    }

    /// One random point matching the filter, used to seed a walk.
    async fn random_point(
        &self,
        args: &Args,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Option<PointId>, anyhow::Error> {
        let mut request_builder = QueryPointsBuilder::new(self.args.collection_name.clone())
            .query(Query::new_sample(Sample::Random))
            .limit(1)
            .with_payload(false)
            .with_vectors(false);

        if let Some(filter) = filter {
            request_builder = request_builder.filter(filter);
        }

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        if let Some(shard_key) = &self.args.shard_key {
            request_builder =
                request_builder.shard_key_selector(vec![Key::Keyword(shard_key.clone())]);
        }

        if let Some(timeout) = self.args.timeout {
            request_builder = request_builder.timeout(timeout as u64);
        }

        let request = request_builder.build();
        let res =
            retry_with_clients(&self.clients, args, |client| client.query(request.clone())).await?;

        Ok(res.result.first().and_then(|point| point.id.clone()))
    }

    /// Check out a walk in progress, or open a new one at a random point. Walks start
    /// at random offsets so that concurrent ones cover different stretches of the
    /// collection: seeded from the top they would all re-read the same pages.
    async fn acquire_walk(
        &self,
        args: &Args,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<(Option<qdrant_client::qdrant::Filter>, Option<PointId>), anyhow::Error> {
        // Bind before the `if let` so the guard is dropped before the await below.
        let resumed = self.walks.lock().unwrap().pop();
        if let Some(walk) = resumed {
            // A walk in progress keeps its own filter: the cursor only means
            // anything against the query that produced it.
            return Ok((walk.filter, Some(walk.offset)));
        }

        let offset = self.random_point(args, filter.clone()).await?;
        Ok((filter, offset))
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

        let mut rng = rand::rng();
        let filter = self.build_filter(&mut rng, args);

        // Opening a walk costs a seeding query. That is setup, so it is resolved
        // before the timer starts rather than charged to the request.
        let walk = match self.mode {
            ScrollMode::Sequential => Some(self.acquire_walk(args, filter.clone()).await?),
            _ => None,
        };

        let start = std::time::Instant::now();

        let (result_len, server_time) = match self.mode {
            ScrollMode::Sample => self.sample(args, filter).await?,
            ScrollMode::Scroll => {
                let (len, time, _) = self.scroll_cursor(args, filter, None).await?;
                (len, time)
            }
            ScrollMode::Sequential => {
                let (filter, offset) = walk.expect("sequential acquires its walk above");

                let (len, time, next) = self.scroll_cursor(args, filter.clone(), offset).await?;
                // Return the walk advanced. A walk that ran off the end is not returned:
                // the next request opens a fresh one elsewhere instead of replaying it.
                if let Some(offset) = next {
                    self.walks.lock().unwrap().push(Walk { filter, offset });
                }
                (len, time)
            }
        };

        let elapsed = start.elapsed().as_secs_f32();
        let delay_millis = self.start_time.elapsed().as_millis() as u32;
        let full_timing = Timing {
            delay_millis,
            value: elapsed,
        };

        if server_time > self.args.timing_threshold {
            progress_bar.println(format!("Slow scroll: {server_time:?}"));
        }

        if result_len < self.args.search_limit {
            progress_bar.println(format!(
                "Scroll result is too small: {result_len} of {}",
                self.args.search_limit
            ));
        }

        let server_timing = Timing {
            delay_millis,
            value: server_time as f32,
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
