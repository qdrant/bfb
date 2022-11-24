use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use anyhow::Error;
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{PointId, PointStruct};
use rand::Rng;
use crate::{Args, random_payload, random_vector};

pub struct UpsertProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    client: QdrantClient,
    progress_bar: Arc<ProgressBar>
}


impl UpsertProcessor {
    pub fn new(args: Args, stopped: Arc<AtomicBool>, client: QdrantClient, progress_bar: Arc<ProgressBar>) -> Self {
        UpsertProcessor { args, stopped, client, progress_bar }
    }

    pub async fn upsert(&self, batch_id: usize) -> Result<(), anyhow::Error> {
        let points_uploaded = self.args.batch_size * batch_id;
        let points_left = self.args.num_vectors.saturating_sub(points_uploaded);

        if points_left == 0 {
            return Ok(());
        }

        let mut rng = rand::thread_rng();
        let mut points = Vec::new();
        for i in 0..min(self.args.batch_size, points_left) {
            let idx = if let Some(max_id) = self.args.max_id {
                rng.gen_range(0..max_id) as u64
            } else {
                batch_id as u64 * self.args.batch_size as u64 + i as u64
            };

            let point_id: PointId = PointId {
                point_id_options: Some(if self.args.uuids {
                    PointIdOptions::Uuid(uuid::Uuid::from_u128(idx as u128).to_string())
                } else {
                    PointIdOptions::Num(idx)
                })
            };

            points.push(PointStruct::new(
                point_id,
                random_vector(self.args.dim),
                random_payload(self.args.keywords),
            ));
        }

        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let res = if self.args.wait_on_upsert {
            self.client.upsert_points_blocking(&self.args.collection_name, points).await?
        } else {
            self.client.upsert_points(&self.args.collection_name, points).await?
        };
        if res.time > self.args.timing_threshold {
            self.progress_bar.println(format!("Slow upsert: {:?}", res.time));
        }
        Ok::<(), Error>(())
    }
}
