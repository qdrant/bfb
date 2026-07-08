use crate::args::Args;
use crate::common::{
    DEFAULT_VOCAB_SIZE, Timing, create_zipf, random_dense_vector, random_filter,
    random_sparse_vector, random_vector_name, retry_with_clients,
};
use crate::processor::Processor;
use crate::search_config::SearchConfig;
use crate::search_generator::ConfigSearchGenerator;
use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    BatchResult, PrefetchQueryBuilder, QuantizationSearchParamsBuilder, Query,
    QueryBatchPointsBuilder, QueryPointsBuilder, ScoredPoint, SearchParamsBuilder, SparseIndices,
    VectorInput,
};
use rand::Rng;
use rand::RngExt;

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct SearchStats {
    server_timings: Vec<Timing>,
    qps: Vec<Timing>,
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

/// Recall of a single search against reference-dataset ground truth:
/// `|returned_ids ∩ expected[:top]| / top`, where `top` is bounded by both the
/// search limit and the number of ground-truth neighbors available. Ground
/// truth ids are dataset corpus indices, which match integer point ids when the
/// dataset was uploaded with the default integer id scheme.
fn recall_against_ground_truth(
    returned: &[ScoredPoint],
    expected: &[u64],
    search_limit: usize,
) -> f32 {
    let top = search_limit.min(expected.len());
    if top == 0 {
        return 1.0;
    }

    let expected_top: HashSet<u64> = expected.iter().take(top).copied().collect();

    let found = returned
        .iter()
        .filter_map(|p| match p.id.as_ref()?.point_id_options.as_ref()? {
            PointIdOptions::Num(num) => Some(*num),
            PointIdOptions::Uuid(_) => None,
        })
        .filter(|id| expected_top.contains(id))
        .count();

    found as f32 / top as f32
}

fn compare_batch_search_results(
    exact_search: &[BatchResult],
    normal_search: &[BatchResult],
) -> Vec<f32> {
    assert_eq!(exact_search.len(), normal_search.len());
    exact_search
        .iter()
        .zip(normal_search)
        .map(|(exact, normal)| compare_search_results(&exact.result, &normal.result))
        .collect()
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

/// YAML-config-driven search (`bfb search --file config.yaml`).
pub struct ConfigSearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    stats: Mutex<SearchStats>,
    generator: ConfigSearchGenerator,
}

impl ConfigSearchProcessor {
    pub fn new(
        args: Args,
        config: &SearchConfig,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
    ) -> anyhow::Result<Self> {
        Ok(ConfigSearchProcessor {
            args: args.clone(),
            stopped,
            clients,
            start_timestamp_millis: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as f64,
            start_time: std::time::Instant::now(),
            stats: Mutex::new(SearchStats::default()),
            generator: ConfigSearchGenerator::new(config)?,
        })
    }

    fn create_request_builder(
        &self,
        query_filter: Option<qdrant_client::qdrant::Filter>,
        query_vectors: Vec<f32>,
        sparse_indices: Option<SparseIndices>,
        vector_name: Option<String>,
        search_params: SearchParamsBuilder,
    ) -> QueryPointsBuilder {
        let mut request_builder = QueryPointsBuilder::new(self.args.collection_name.clone())
            .with_payload(self.args.search_with_payload)
            .with_vectors(self.args.search_with_vectors)
            .limit(self.args.search_limit as u64);

        if let Some(vector_name) = vector_name.filter(|n| !n.is_empty()) {
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
        req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let start = std::time::Instant::now();
        let mut rng = rand::rng();

        let template_idx = self.generator.random_template_idx(&mut rng);

        let mut quantization_params_builder = QuantizationSearchParamsBuilder::default()
            .rescore(self.args.quantization_rescore.unwrap_or_default());

        if let Some(oversampling) = self.args.quantization_oversampling {
            quantization_params_builder = quantization_params_builder.oversampling(oversampling);
        }

        let mut search_params = SearchParamsBuilder::default()
            .exact(self.args.search_exact && !self.args.search_quality)
            .quantization(quantization_params_builder)
            .indexed_only(self.args.indexed_only.unwrap_or_default());

        if let Some(hnsw_ef) = self.args.search_hnsw_ef {
            search_params = search_params.hnsw_ef(hnsw_ef as u64);
        }

        // Materialize the whole batch up front so each query keeps its own
        // filter and (for dataset query sources) its ground-truth ids.
        let generated: Vec<_> = (0..self.args.search_batch_size)
            .map(|_| {
                self.generator
                    .make_query_for(template_idx, req_id, &mut rng)
            })
            .collect();

        // Ground-truth ids present ⇒ measure recall against the dataset instead
        // of the exact-search self-comparison.
        let expected_ids: Vec<Option<Vec<u64>>> =
            generated.iter().map(|g| g.expected_ids.clone()).collect();
        let has_ground_truth = expected_ids.iter().any(Option::is_some);

        let query_points: Vec<_> = generated
            .into_iter()
            .map(|generated| {
                let query_filter = generated.filter.clone();
                let (query_vectors, sparse_indices, vector_name) =
                    if let Some((values, indices, name)) = generated.sparse {
                        (values, Some(indices), Some(name))
                    } else if let Some((values, name)) = generated.dense {
                        (values, None, name)
                    } else {
                        panic!("search config request must produce a dense or sparse vector");
                    };

                self.create_request_builder(
                    query_filter,
                    query_vectors,
                    sparse_indices,
                    vector_name,
                    search_params.clone(),
                )
                .build()
            })
            .collect();

        // Exact-search quality comparison only makes sense without dataset
        // ground truth; otherwise we score directly against the known answers.
        let (query_points_for_quality, batch_query_points) =
            if self.args.search_quality && !has_ground_truth {
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

        // Reference-dataset accuracy: compare each query's returned ids to the
        // dataset's ground-truth neighbors (recall@k), matching
        // vector-db-benchmark's `|found ∩ expected[:k]| / k`.
        if has_ground_truth {
            let recalls: Vec<f32> = res_batch
                .result
                .iter()
                .zip(&expected_ids)
                .filter_map(|(result, expected)| {
                    expected.as_ref().map(|expected| {
                        recall_against_ground_truth(
                            &result.result,
                            expected,
                            self.args.search_limit,
                        )
                    })
                })
                .collect();
            let mut stats_guard = self.stats.lock().unwrap();
            stats_guard.precisions.extend(recalls);
            return Ok(());
        }

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

impl Processor for ConfigSearchProcessor {
    async fn make_request(
        &self,
        req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        self.search(req_id, args, progress_bar).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::PointId as GrpcPointId;

    fn scored(ids: &[u64]) -> Vec<ScoredPoint> {
        ids.iter()
            .map(|&id| ScoredPoint {
                id: Some(GrpcPointId::from(id)),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn recall_perfect_and_partial() {
        // All top-3 found.
        let returned = scored(&[1, 2, 3]);
        assert_eq!(
            recall_against_ground_truth(&returned, &[1, 2, 3, 4], 3),
            1.0
        );

        // 2 of top-3 found.
        let returned = scored(&[1, 9, 3]);
        assert!((recall_against_ground_truth(&returned, &[1, 2, 3], 3) - 2.0 / 3.0).abs() < 1e-6);

        // Ground truth shorter than the limit ⇒ denominator is bounded by it.
        let returned = scored(&[5, 6]);
        assert_eq!(recall_against_ground_truth(&returned, &[5, 6], 10), 1.0);
    }

    #[test]
    fn recall_ignores_extra_returned_ids() {
        // Returning more than `top` correct ids never exceeds 1.0.
        let returned = scored(&[1, 2, 3, 4, 5]);
        assert_eq!(recall_against_ground_truth(&returned, &[1, 2], 2), 1.0);
    }
}
