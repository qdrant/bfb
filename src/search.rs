use crate::common::{random_vector_name, retry_with_clients};
use crate::{random_filter, random_vector, Args};
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::{QuantizationSearchParams, SearchParams, SearchPoints};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<QdrantClient>,
    pub server_timings: Mutex<Vec<f64>>,
    pub full_timings: Mutex<Vec<f64>>,
}

impl SearchProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, clients: Vec<QdrantClient>) -> Self {
        SearchProcessor {
            args,
            stopped,
            clients,
            server_timings: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
        }
    }

    pub async fn search(
        &self,
        _req_id: usize,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let query_vector = random_vector(self.args.dim);
        let query_filter = random_filter(self.args.keywords);

        let start = std::time::Instant::now();

        let vector_name = if self.args.vectors_per_point > 1 {
            Some(random_vector_name(self.args.vectors_per_point))
        } else {
            None
        };

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
                ..Default::default()
            }),
            score_threshold: None,
            offset: None,
            vector_name,
            with_vectors: None,
            read_consistency: self.args.read_consistency.map(Into::into),
        };

        let res =
            retry_with_clients(&self.clients, |client| client.search_points(&request)).await?;

        let elapsed = start.elapsed().as_secs_f64();

        self.full_timings.lock().unwrap().push(elapsed);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res.time));
        }
        self.server_timings.lock().unwrap().push(res.time);
        Ok(())
    }
}
