use crate::fbin_reader::FBinReader;
use crate::{random_payload, random_vector, Args};
use anyhow::Error;
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{PointId, PointStruct, Vectors};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::cmp::min;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct UpsertProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<QdrantClient>,
    progress_bar: Arc<ProgressBar>,
    reader: Option<FBinReader>,
}

impl UpsertProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<QdrantClient>,
        progress_bar: Arc<ProgressBar>,
        reader: Option<FBinReader>,
    ) -> Self {
        UpsertProcessor {
            args,
            stopped,
            clients,
            progress_bar,
            reader,
        }
    }

    pub async fn upsert(&self, batch_id: usize) -> Result<(), anyhow::Error> {
        let points_uploaded = self.args.batch_size * batch_id;
        let points_left = self.args.num_vectors.saturating_sub(points_uploaded);

        if points_left == 0 {
            return Ok(());
        }

        let mut rng = rand::thread_rng();
        let mut points = Vec::new();

        let mut batch_ids = Vec::new();

        for i in 0..min(self.args.batch_size, points_left) {
            let idx = if let Some(max_id) = self.args.max_id {
                rng.gen_range(0..max_id) as u64
            } else {
                batch_id as u64 * self.args.batch_size as u64 + i as u64
            };

            let point_id: PointId = PointId {
                point_id_options: Some(if self.args.uuids {
                    let random_uuid = uuid::Uuid::new_v4();
                    PointIdOptions::Uuid(random_uuid.to_string())
                } else {
                    PointIdOptions::Num(idx)
                }),
            };

            batch_ids.push(point_id.clone());

            let vectors: Vectors = if let Some(reader) = &self.reader {
                reader.read_vector(idx as usize).to_vec().into()
            } else if self.args.vectors_per_point > 1 {
                let vectors_map: HashMap<_, _> = (0..self.args.vectors_per_point)
                    .map(|i| {
                        let vector_name = format!("{}", i);
                        let vector = random_vector(self.args.dim);
                        (vector_name, vector)
                    })
                    .collect();
                vectors_map.into()
            } else {
                random_vector(self.args.dim).into()
            };

            points.push(PointStruct::new(
                point_id,
                vectors,
                random_payload(self.args.keywords),
            ));
        }

        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let ordering = self.args.write_ordering.map(Into::into);

        let res = if self.args.wait_on_upsert {
            self.clients
                .choose(&mut rng)
                .unwrap()
                .upsert_points_blocking(&self.args.collection_name, points, ordering.clone())
                .await?
        } else {
            self.clients
                .choose(&mut rng)
                .unwrap()
                .upsert_points(&self.args.collection_name, points, ordering.clone())
                .await?
        };

        if self.args.set_payload {
            if self.args.wait_on_upsert {
                self.clients
                    .choose(&mut rng)
                    .unwrap()
                    .set_payload_blocking(
                        &self.args.collection_name,
                        &batch_ids.into(),
                        random_payload(self.args.keywords),
                        ordering,
                    )
                    .await?;
            } else {
                self.clients
                    .choose(&mut rng)
                    .unwrap()
                    .set_payload(
                        &self.args.collection_name,
                        &batch_ids.into(),
                        random_payload(self.args.keywords),
                        ordering,
                    )
                    .await?;
            }
        }

        if res.time > self.args.timing_threshold {
            self.progress_bar
                .println(format!("Slow upsert: {:?}", res.time));
        }
        Ok::<(), Error>(())
    }
}
