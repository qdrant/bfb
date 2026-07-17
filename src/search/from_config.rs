//! YAML-config-driven search benchmarking (`bfb search --file config.yaml`).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::{
    PrefetchQueryBuilder, QuantizationSearchParamsBuilder, Query, QueryBatchPointsBuilder,
    QueryPointsBuilder, SearchParamsBuilder, SparseIndices, VectorInput,
};

use super::{SearchStats, compare_batch_search_results, recall_against_ground_truth};
use crate::args::Args;
use crate::client::retry_with_clients;
use crate::config::search::SearchConfig;
use crate::generators::ConfigSearchGenerator;
use crate::processor::{Processor, Timing};

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

        if let Some(shard_key) = &self.args.shard_key {
            request_builder =
                request_builder.shard_key_selector(vec![Key::Keyword(shard_key.clone())]);
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
