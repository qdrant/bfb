use crate::common::{random_sparse_vector, random_vector_name, retry_with_clients, Timing};
use crate::processor::Processor;
use crate::{random_dense_vector, random_filter, Args};
use indicatif::ProgressBar;
use qdrant_client::qdrant::{
    QuantizationSearchParamsBuilder, SearchParamsBuilder, SearchPointsBuilder, SparseIndices,
    Vector,
};
use qdrant_client::Qdrant;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    pub start_timestamp_millis: f64,
    start_time: std::time::Instant,
    pub server_timings: Mutex<Vec<Timing>>,
    pub rps: Mutex<Vec<Timing>>,
    pub full_timings: Mutex<Vec<Timing>>,
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
            server_timings: Mutex::new(Vec::new()),
            rps: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
            uuids,
        }
    }

    fn get_sparse_query(&self) -> (Vec<f32>, Option<SparseIndices>, Option<String>) {
        if let Some(sparsity) = self.args.sparse_vectors {
            let sparse_vector: Vector =
                random_sparse_vector(self.args.sparse_dim.unwrap_or(self.args.dim), sparsity)
                    .into();
            let query_vector = sparse_vector.data;
            let sparse_indices = sparse_vector.indices;
            let name = format!(
                "{}_sparse",
                random_vector_name(self.args.sparse_vectors_per_point)
            );
            (query_vector, sparse_indices, Some(name))
        } else {
            panic!("No sparse vectors configured")
        }
    }

    fn get_dense_query(&self) -> (Vec<f32>, Option<SparseIndices>, Option<String>) {
        let query_vector = random_dense_vector(self.args.dim);
        let sparse_indices = None;
        if self.args.vectors_per_point > 1 {
            let name = random_vector_name(self.args.vectors_per_point);
            (query_vector, sparse_indices, Some(name))
        } else {
            (query_vector, sparse_indices, None)
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
            self.args.keywords.first().cloned(),
            self.args.float_payloads.first().cloned().unwrap_or(false),
            self.args.int_payloads.first().cloned(),
            &self.uuids,
            self.args.match_any,
        );

        let mut request_builder = SearchPointsBuilder::new(
            self.args.collection_name.clone(),
            query_vector,
            self.args.search_limit as u64,
        )
        .with_payload(self.args.search_with_payload);

        if let Some(vector_name) = vector_name {
            request_builder = request_builder.vector_name(vector_name);
        }

        if let Some(filter) = query_filter {
            request_builder = request_builder.filter(filter);
        }

        if let Some(sparse_indices) = sparse_indices {
            request_builder = request_builder.sparse_indices(sparse_indices);
        }

        let mut quantization_params_builder = QuantizationSearchParamsBuilder::default()
            .rescore(self.args.quantization_rescore.unwrap_or_default());

        if let Some(oversampling) = self.args.quantization_oversampling {
            quantization_params_builder = quantization_params_builder.oversampling(oversampling);
        }

        let mut search_params = SearchParamsBuilder::default()
            .exact(self.args.search_exact)
            .quantization(quantization_params_builder)
            .indexed_only(self.args.indexed_only.unwrap_or_default());

        if let Some(hnsw_ef) = self.args.hnsw_ef_construct {
            search_params = search_params.hnsw_ef(hnsw_ef as u64);
        }

        if let Some(read_cosistency) = self.args.read_consistency {
            request_builder = request_builder.read_consistency(read_cosistency);
        }

        request_builder = request_builder.params(search_params);

        let request = request_builder.build();
        let res = retry_with_clients(&self.clients, args, |client| {
            client.search_points(request.clone())
        })
        .await?;

        let elapsed = start.elapsed().as_secs_f64();

        let full_timing = Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: elapsed,
        };

        self.full_timings.lock().unwrap().push(full_timing);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res.time));
        }

        if res.result.len() < self.args.search_limit && self.args.uuid_payloads.is_empty() {
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

        self.server_timings.lock().unwrap().push(server_timing);
        self.rps.lock().unwrap().push(rps_timing);

        if let Some(delay_millis) = self.args.delay {
            tokio::time::sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
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
        self.server_timings.lock().unwrap().clone()
    }

    fn rps(&self) -> Vec<Timing> {
        self.rps.lock().unwrap().clone()
    }

    fn full_timings(&self) -> Vec<Timing> {
        self.full_timings.lock().unwrap().clone()
    }
}
