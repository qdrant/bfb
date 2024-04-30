use crate::common::{random_sparse_vector, random_vector_name, retry_with_clients};
use crate::{random_dense_vector, random_filter, Args};
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::{
    QuantizationSearchParams, SearchParams, SearchPoints, SparseIndices, Vector,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<QdrantClient>,
    pub server_timings: Mutex<Vec<f64>>,
    pub rps: Mutex<Vec<f64>>,
    pub full_timings: Mutex<Vec<f64>>,
}

impl SearchProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, clients: Vec<QdrantClient>) -> Self {
        SearchProcessor {
            args,
            stopped,
            clients,
            server_timings: Mutex::new(Vec::new()),
            rps: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
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
            self.args.match_any,
        );

        let request = SearchPoints {
            collection_name: self.args.collection_name.to_string(),
            vector: query_vector,
            filter: query_filter,
            limit: self.args.search_limit as u64,
            with_payload: Some(self.args.search_with_payload.into()),
            params: Some(SearchParams {
                hnsw_ef: self.args.search_hnsw_ef.map(|v| v as u64),
                quantization: Some(QuantizationSearchParams {
                    ignore: None,
                    rescore: self.args.quantization_rescore,
                    oversampling: self.args.quantization_oversampling,
                }),
                indexed_only: self.args.indexed_only,
                ..Default::default()
            }),
            score_threshold: None,
            offset: None,
            vector_name,
            with_vectors: None,
            read_consistency: self.args.read_consistency.map(Into::into),
            timeout: None,
            shard_key_selector: None,
            sparse_indices,
        };

        let res = retry_with_clients(&self.clients, args, |client| client.search_points(&request))
            .await?;

        let elapsed = start.elapsed().as_secs_f64();

        self.full_timings.lock().unwrap().push(elapsed);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res.time));
        }

        if res.result.len() < self.args.search_limit {
            progress_bar.println(format!(
                "Search result is too small: {} of {}",
                res.result.len(),
                self.args.search_limit
            ));
        }

        self.server_timings.lock().unwrap().push(res.time);
        self.rps.lock().unwrap().push(progress_bar.per_sec());

        if let Some(delay_millis) = self.args.delay {
            tokio::time::sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        Ok(())
    }
}
