use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicBool;
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::SearchPoints;
use crate::{Args, random_filter, random_vector};

pub struct SearchProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    client: QdrantClient,
    pub server_timings: Mutex<Vec<f64>>,
    pub full_timings: Mutex<Vec<f64>>,
}


impl SearchProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, client: QdrantClient) -> Self {
        SearchProcessor {
            args,
            stopped,
            client,
            server_timings: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
        }
    }

    pub async fn search(&self, _req_id: usize, progress_bar: &ProgressBar) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let query_vector = random_vector(self.args.dim);
        let query_filter = random_filter(self.args.keywords);

        let start = std::time::Instant::now();

        let res = self.client.search_points(&SearchPoints {
            collection_name: self.args.collection_name.to_string(),
            vector: query_vector,
            filter: query_filter,
            limit: self.args.search_limit as u64,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            vector_name: None,
            with_vectors: None,
        }).await?;
        let elapsed = start.elapsed().as_secs_f64();

        self.full_timings.lock().unwrap().push(elapsed);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow search: {:?}", res.time));
        }
        self.server_timings.lock().unwrap().push(res.time);
        Ok(())
    }
}