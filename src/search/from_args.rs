//! Flag-driven search benchmarking (legacy CLI path).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    PrefetchQueryBuilder, QuantizationSearchParamsBuilder, Query, QueryBatchPointsBuilder,
    QueryPointsBuilder, SearchParamsBuilder, SparseIndices, VectorInput,
};
use rand::Rng;
use rand::RngExt;

use super::{SearchStats, compare_batch_search_results};
use crate::args::Args;
use crate::client::retry_with_clients;
use crate::generators::random::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_dense_vector, random_filter, random_sparse_vector,
    random_vector_name,
};
use crate::processor::{Processor, Timing};

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    stats: Mutex<SearchStats>,
    pub uuids: Vec<String>,
    zipf: Option<rand_distr::Zipf<f64>>,
}

impl SearchProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        uuids: Vec<String>,
    ) -> Self {
        let zipf = args
            .text_payloads
            .then(|| create_zipf(args.text_payload_vocabulary.unwrap_or(DEFAULT_VOCAB_SIZE)));

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
            zipf,
        }
    }

    fn get_sparse_queries(
        &self,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<f32>, Option<SparseIndices>, Option<String>)> {
        if let Some(sparsity) = self.args.sparse_vectors {
            let name = format!(
                "{}_sparse",
                random_vector_name(rng, self.args.sparse_vectors_per_point)
            );

            (0..self.args.search_batch_size)
                .map(|_| {
                    let vocab_size = self.args.sparse_dim.unwrap_or(self.args.dim);
                    let length = ((vocab_size as f64) * sparsity).ceil() as usize;
                    let sparse_vector_tuples = random_sparse_vector(rng, vocab_size, length);
                    let (indices, values): (Vec<_>, Vec<_>) =
                        sparse_vector_tuples.into_iter().unzip();
                    let sparse_indices = SparseIndices { data: indices };
                    (values, Some(sparse_indices), Some(name.clone()))
                })
                .collect()
        } else {
            panic!("No sparse vectors configured")
        }
    }

    fn get_dense_queries(
        &self,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<f32>, Option<SparseIndices>, Option<String>)> {
        let name = if self.args.vectors_per_point > 1 {
            let name = random_vector_name(rng, self.args.vectors_per_point);
            Some(name)
        } else {
            None
        };

        let is_uint = self.args.is_uint8_datatype();

        (0..self.args.search_batch_size)
            .map(|_| {
                (
                    random_dense_vector(rng, self.args.dim, is_uint),
                    None,
                    name.clone(),
                )
            })
            .collect()
    }

    fn create_request_builder(
        &self,
        query_filter: Option<qdrant_client::qdrant::Filter>,
        query_vectors: Vec<f32>,
        sparse_indices: Option<SparseIndices>,
        vector_name: Option<String>,
        search_params: SearchParamsBuilder,
    ) -> QueryPointsBuilder {
        let mut request_builder = QueryPointsBuilder::new(
            self.args.collection_name.clone(),
            // query_vector,
        )
        .with_payload(self.args.search_with_payload)
        .with_vectors(self.args.search_with_vectors)
        .limit(self.args.search_limit as u64);

        if let Some(vector_name) = vector_name {
            request_builder = request_builder.using(vector_name);
        }

        if let Some(filter) = query_filter {
            request_builder = request_builder.filter(filter);
        }

        let vector = if let Some(sparse_indices) = sparse_indices {
            VectorInput::new_sparse(sparse_indices.data, query_vectors)
        } else {
            VectorInput::new_dense(query_vectors)
        };

        let query = Query::new_nearest(vector);

        if let Some(prefetch_limit) = self.args.prefetch {
            let mut prefetch_params = SearchParamsBuilder::default()
                .quantization(QuantizationSearchParamsBuilder::default().rescore(false));

            if let Some(hnsw_ef) = self.args.search_hnsw_ef {
                prefetch_params = prefetch_params.hnsw_ef(hnsw_ef as u64);
            }

            let prefetch = PrefetchQueryBuilder::default()
                .query(query.clone())
                .params(prefetch_params)
                .limit(prefetch_limit as u64)
                .build();

            request_builder = request_builder.prefetch(vec![prefetch]);
            request_builder = request_builder.params(search_params);
        } else {
            request_builder = request_builder.params(search_params);
        }

        request_builder = request_builder.query(query);

        if let Some(read_consistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_consistency);
        }

        request_builder
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
        let mut rng = rand::rng();
        let has_sparse = self.args.sparse_vectors.is_some();
        let has_dense = self.args.vectors_per_point > 0;

        let use_sparse = match (has_sparse, has_dense) {
            (true, true) => rng.random_bool(0.5),
            (true, false) => true,
            (false, true) => false,
            (false, false) => panic!("No sparse or dense vectors"),
        };

        let query_batch = if use_sparse {
            self.get_sparse_queries(&mut rng)
        } else {
            self.get_dense_queries(&mut rng)
        };

        let query_filter = random_filter(
            &mut rng,
            &self.args.keywords,
            &self.args.float_payloads,
            &self.args.int_payloads,
            &self.uuids,
            self.args.match_any,
            self.args.geo_payloads,
            self.args.bool_payloads,
            self.args.keywords_length_multiplier,
            self.zipf.as_ref(),
        );

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

        if let Some(hnsw_ef) = self.args.search_hnsw_ef {
            search_params = search_params.hnsw_ef(hnsw_ef as u64);
        }

        let query_points: Vec<_> = query_batch
            .into_iter()
            .map(|(query_vectors, sparse_indices, vector_name)| {
                self.create_request_builder(
                    query_filter.clone(),
                    query_vectors,
                    sparse_indices,
                    vector_name,
                    search_params.clone(),
                )
                .build()
            })
            .collect();

        let (query_points_for_quality, batch_query_points) = if self.args.search_quality {
            (Some(query_points.clone()), query_points)
        } else {
            (None, query_points)
        };

        let mut batch_request_builder =
            QueryBatchPointsBuilder::new(self.args.collection_name.clone(), batch_query_points);
        if let Some(timeout) = self.args.timeout {
            batch_request_builder = batch_request_builder.timeout(timeout as u64);
        }

        let request = batch_request_builder.build();
        let res_batch = retry_with_clients(&self.clients, args, |client| {
            client.query_batch(request.clone())
        })
        .await?;

        let elapsed = start.elapsed().as_secs_f32();
        let delay_millis = self.start_time.elapsed().as_millis() as u32;

        let full_timing = Timing {
            delay_millis,
            value: elapsed,
        };

        if res_batch.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res_batch.time));
        }

        let server_timing = Timing {
            delay_millis,
            value: res_batch.time as f32,
        };

        let qps_timing = Timing {
            delay_millis,
            value: progress_bar.per_sec() as f32,
        };

        let rps_timing = Timing {
            delay_millis,
            value: (progress_bar.per_sec() / self.args.search_batch_size as f64) as f32,
        };

        // Update stats all at once
        {
            let mut stats_guard = self.stats.lock().unwrap();
            stats_guard.server_timings.push(server_timing);
            stats_guard.qps.push(qps_timing);
            stats_guard.rps.push(rps_timing);
            stats_guard.full_timings.push(full_timing);
        }

        if let Some(delay_millis) = self.args.delay {
            tokio::time::sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        // Search quality bench

        let Some(mut exact_query_points) = query_points_for_quality else {
            return Ok(());
        };

        let exact_search = search_params.clone().exact(true).build();
        for point in &mut exact_query_points {
            point.params = Some(exact_search);
        }
        let mut exact_request_builder =
            QueryBatchPointsBuilder::new(self.args.collection_name.clone(), exact_query_points);
        if let Some(timeout) = self.args.timeout {
            exact_request_builder = exact_request_builder.timeout(timeout as u64);
        }
        let exact_request = exact_request_builder.build();

        let exact_res = retry_with_clients(&self.clients, args, |client| {
            client.query_batch(exact_request.clone())
        })
        .await?;

        let precisions = compare_batch_search_results(&res_batch.result, &exact_res.result);

        {
            let mut stats_guard = self.stats.lock().unwrap();
            stats_guard.precisions.extend(precisions);
        }

        Ok(())
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

    fn qps(&self) -> Vec<Timing> {
        self.stats.lock().unwrap().qps.clone()
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

    fn get_batch_size(&self) -> usize {
        self.args.search_batch_size
    }
}
