use crate::{random_filter, Args};
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::ScrollPoints;
use rand::prelude::SliceRandom;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct ScrollProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<QdrantClient>,
    pub server_timings: Mutex<Vec<f64>>,
    pub full_timings: Mutex<Vec<f64>>,
}

impl ScrollProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, clients: Vec<QdrantClient>) -> Self {
        ScrollProcessor {
            args,
            stopped,
            clients,
            server_timings: Mutex::new(Vec::new()),
            full_timings: Mutex::new(Vec::new()),
        }
    }

    pub async fn scroll(
        &self,
        offset: u64,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let query_filter = random_filter(self.args.keywords);

        let start = std::time::Instant::now();

        let res = self
            .clients
            .choose(&mut rand::thread_rng())
            .unwrap()
            .scroll(&ScrollPoints {
                collection_name: self.args.collection_name.to_string(),
                filter: query_filter,
                limit: self.args.scroll_limit.map(|v| v as u32),
                with_payload: Some(self.args.search_with_payload.into()),
                offset: Some(offset.into()),
                with_vectors: None,
                read_consistency: self.args.read_consistency.map(Into::into),
            })
            .await?;
        let elapsed = start.elapsed().as_secs_f64();

        self.full_timings.lock().unwrap().push(elapsed);

        if res.time > self.args.timing_threshold {
            progress_bar.println(format!("Slow scroll: {:?}", res.time));
        }
        self.server_timings.lock().unwrap().push(res.time);
        Ok(())
    }
}
