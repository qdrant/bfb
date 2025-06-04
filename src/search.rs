use crate::common::{random_sparse_vector, random_vector_name, retry_with_clients, Timing};
use crate::processor::Processor;
use crate::{random_dense_vector, random_filter, Args};
use indicatif::ProgressBar;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    PrefetchQueryBuilder, QuantizationSearchParamsBuilder, Query, QueryPointsBuilder, ScoredPoint,
    SearchParamsBuilder, SparseIndices, VectorInput,
};
use qdrant_client::Qdrant;

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct SearchStats {
    server_timings: Vec<Timing>,
    rps: Vec<Timing>,
    full_timings: Vec<Timing>,
    precisions: Vec<f32>,
}

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    stats: Mutex<SearchStats>,
    pub uuids: Vec<String>,
}

impl SearchProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        uuids: Vec<String>,
    ) -> Self {
        SearchProcessor {
            args,
            stopped,
            clients,
            start_timestamp_millis: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as f64,
            start_time: std::time::Instant::now(),
            stats: Mutex::new(SearchStats::default()),
            uuids,
        }
    }

    fn get_sparse_query(&self) -> (Vec<f32>, Option<SparseIndices>, Option<String>) {
        if let Some(sparsity) = self.args.sparse_vectors {
            let sparse_vector_tuples =
                random_sparse_vector(self.args.sparse_dim.unwrap_or(self.args.dim), sparsity);
            let (indices, values): (Vec<_>, Vec<_>) = sparse_vector_tuples.into_iter().unzip();
            let sparse_indices = SparseIndices { data: indices };
            let name = format!(
                "{}_sparse",
                random_vector_name(self.args.sparse_vectors_per_point)
            );
            (values, Some(sparse_indices), Some(name))
        } else {
            panic!("No sparse vectors configured")
        }
    }

    fn get_dense_query(&self) -> (Vec<f32>, Option<SparseIndices>, Option<String>) {
        let query_vector = random_dense_vector(self.args.dim);
        if self.args.vectors_per_point > 1 {
            let name = random_vector_name(self.args.vectors_per_point);
            (query_vector, None, Some(name))
        } else {
            (query_vector, None, None)
        }
    }

    pub async fn search(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let start = std::time::Instant::now();

        let has_sparse = self.args.sparse_vectors.is_some();
        let has_dense = self.args.vectors_per_point > 0;

        let use_sparse = match (has_sparse, has_dense) {
            (true, true) => rand::random::<bool>(),
            (true, false) => true,
            (false, true) => false,
            (false, false) => panic!("No sparse or dense vectors"),
        };

        let (query_vector, sparse_indices, vector_name) = if use_sparse {
            self.get_sparse_query()
        } else {
            self.get_dense_query()
        };

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
                    .unwrap_or(crate::common::DEFAULT_VOCAB_SIZE)
            }),
        );

        let mut request_builder = QueryPointsBuilder::new(
            self.args.collection_name.clone(),
            // query_vector,
        )
        .with_payload(self.args.search_with_payload)
        .limit(self.args.search_limit as u64);

        if let Some(vector_name) = vector_name {
            request_builder = request_builder.using(vector_name);
        }

        if let Some(filter) = query_filter {
            request_builder = request_builder.filter(filter);
        }

        let vector = if let Some(sparse_indices) = sparse_indices {
            VectorInput::new_sparse(sparse_indices.data, query_vector)
        } else {
            VectorInput::new_dense(query_vector)
        };

        let query = Query::new_nearest(vector);

        if let Some(prefetch_limit) = self.args.prefetch {
            let prefetch_query = PrefetchQueryBuilder::default();

            prefetch_query.query(query.clone());

            let mut params = SearchParamsBuilder::default()
                .quantization(QuantizationSearchParamsBuilder::default().rescore(false));

            if let Some(hnsw_ef) = self.args.hnsw_ef_construct {
                params = params.hnsw_ef(hnsw_ef as u64);
            }

            request_builder = request_builder.params(params);
            request_builder = request_builder.limit(prefetch_limit as u64);
        }

        request_builder = request_builder.query(query);

        let mut quantization_params_builder = QuantizationSearchParamsBuilder::default()
            .rescore(self.args.quantization_rescore.unwrap_or_default());

        if let Some(oversampling) = self.args.quantization_oversampling {
            quantization_params_builder = quantization_params_builder.oversampling(oversampling);
        }

        let mut search_params = SearchParamsBuilder::default()
            // Never do exact search here when measuring search_quality.
            .exact(self.args.search_exact && !self.args.search_quality)
            .quantization(quantization_params_builder)
            .indexed_only(self.args.indexed_only.unwrap_or_default());

        if let Some(hnsw_ef) = self.args.hnsw_ef_construct {
            search_params = search_params.hnsw_ef(hnsw_ef as u64);
        }

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        request_builder = request_builder.params(search_params.clone());

        let request = request_builder.clone().build();
        let res =
            retry_with_clients(&self.clients, args, |client| client.query(request.clone())).await?;

        let elapsed = start.elapsed().as_secs_f64();

        let full_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: elapsed,
        };

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res.time));
        }

        if res.result.len() < self.args.search_limit
            && !self.args.uuid_payloads
            && !self.args.search_quality
        {
            progress_bar.println(format!(
                "Search result is too small: {} of {}",
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

        // Update stats all at once
        {
            let mut stats_guard = self.stats.lock().unwrap();
            stats_guard.server_timings.push(server_timing);
            stats_guard.rps.push(rps_timing);
            stats_guard.full_timings.push(full_timing);
        }

        if let Some(delay_millis) = self.args.delay {
            tokio::time::sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        if !self.args.search_quality {
            return Ok(());
        }

        // Search quality bench

        let exact_search = search_params.clone().exact(true);
        let exact_request_builder = request_builder.clone().params(exact_search);
        let request = exact_request_builder.build();

        let exact_res =
            retry_with_clients(&self.clients, args, |client| client.query(request.clone())).await?;

        let precision = compare_search_results(&res.result, &exact_res.result);

        {
            let mut stats_guard = self.stats.lock().unwrap();
            stats_guard.precisions.push(precision);
        }

        Ok(())
    }
}

fn compare_search_results(exact_search: &[ScoredPoint], normal_search: &[ScoredPoint]) -> f32 {
    if normal_search.is_empty() {
        return 0.0;
    }

    let exact_ids: HashSet<_> = exact_search
        .iter()
        .map(|i| PointId::from(i.id.as_ref().unwrap().point_id_options.clone().unwrap()))
        .collect();

    let normal_ids: HashSet<_> = normal_search
        .iter()
        .map(|i| PointId::from(i.id.as_ref().unwrap().point_id_options.clone().unwrap()))
        .collect();

    let true_positive: usize = normal_ids.iter().filter(|i| exact_ids.contains(i)).count();
    let false_positive: usize = normal_ids.iter().filter(|i| !exact_ids.contains(i)).count();

    true_positive as f32 / (true_positive as f32 + false_positive as f32)
}

// Copy of `qdrant_client::qdrant::PointId` that implements `Eq` and `Hash`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PointId {
    Num(u64),
    Uuid(String),
}

impl From<PointIdOptions> for PointId {
    fn from(value: PointIdOptions) -> Self {
        match value {
            PointIdOptions::Num(i) => PointId::Num(i),
            PointIdOptions::Uuid(i) => PointId::Uuid(i),
        }
    }
}

impl Processor for SearchProcessor {
    async fn make_request(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        self.search(_req_id, args, progress_bar).await
    }

    fn start_timestamp_millis(&self) -> f64 {
        self.start_timestamp_millis
    }

    fn server_timings(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().server_timings.clone()
    }

    fn rps(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().rps.clone()
    }

    fn full_timings(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().full_timings.clone()
    }

    fn precisions(&self) -> Vec<f32> {
        self.stats.lock().unwrap().precisions.clone()
    }
}
