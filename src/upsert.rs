use std::cmp::min;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Error;
use futures::TryFutureExt;
use indicatif::ProgressBar;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{
    PointId, PointStruct, SetPayloadPointsBuilder, UpsertPointsBuilder, Vector, Vectors,
};
use qdrant_client::{Qdrant, QdrantError};
use rand::Rng;
use tokio::sync::RwLock;
use tokio::time::sleep;

use crate::common::{random_sparse_vector, random_vector, retry_with_clients2, Timing};
use crate::fbin_reader::FBinReader;
use crate::save_jsonl::save_timings_as_jsonl;
use crate::{random_dense_vector, random_payload, Args};

fn log_points2(points: Vec<PointStruct>) -> impl FnOnce(QdrantError) -> QdrantError {
    move |e| {
        let mut point_ids = Vec::new();

        for p in &points {
            if let Some(point_id_option) = p.id.clone().unwrap().point_id_options {
                match point_id_option {
                    PointIdOptions::Num(num) => point_ids.push(num.to_string()),
                    PointIdOptions::Uuid(uuid) => point_ids.push(uuid.to_string()),
                }
            }
        }
        tracing::warn!(
            "Failed while upserting. point_ids={:?} error={e}",
            point_ids.join(", "),
        );
        e
    }
}

pub struct UpsertProcessor {
    args: Args,
    stopped: Arc<AtomicBool>,
    clients: Vec<Qdrant>,
    progress_bar: Arc<ProgressBar>,
    reader: Option<FBinReader>,
    start_timestamp_millis: f64,
    start_time: std::time::Instant,
    timings: RwLock<Vec<Timing>>,
}

impl UpsertProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        progress_bar: Arc<ProgressBar>,
        reader: Option<FBinReader>,
    ) -> Self {
        UpsertProcessor {
            args,
            stopped,
            clients,
            progress_bar,
            reader,
            start_timestamp_millis: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as f64,
            start_time: std::time::Instant::now(),
            timings: RwLock::new(Vec::new()),
        }
    }

    pub async fn upsert(&self, batch_id: usize, args: &Args) -> Result<(), Error> {
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
                rng.gen_range(self.args.offset..max_id) as u64
            } else {
                self.args.offset as u64 + (batch_id as u64 * self.args.batch_size as u64 + i as u64)
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
            } else if self.args.vectors_per_point != 1 {
                let vectors_map: HashMap<_, _> = (0..self.args.vectors_per_point)
                    .map(|i| {
                        let vector_name = format!("{}", i);
                        let vector = random_vector(&self.args);
                        (vector_name, vector)
                    })
                    .collect();
                vectors_map.into()
            } else {
                random_dense_vector(self.args.dim).into()
            };

            let vectors: Vectors = if let Some(sparsity) = self.args.sparse_vectors {
                let mut vectors_map: HashMap<_, _> = Default::default();

                for i in 0..self.args.sparse_vectors_per_point {
                    let vector_name = format!("{}_sparse", i);
                    let vector = Vector::from(random_sparse_vector(
                        self.args.sparse_dim.unwrap_or(self.args.dim),
                        sparsity,
                    ));
                    vectors_map.insert(vector_name, vector);
                }

                match vectors.vectors_options {
                    None => {}
                    Some(vectors) => match vectors {
                        VectorsOptions::Vector(vector) => {
                            vectors_map.insert("".to_string(), vector);
                        }
                        VectorsOptions::Vectors(vectors) => {
                            for (name, vector) in vectors.vectors.into_iter() {
                                vectors_map.insert(name, vector);
                            }
                        }
                    },
                }
                vectors_map.into()
            } else {
                vectors
            };

            points.push(PointStruct::new(
                point_id,
                vectors,
                random_payload(&self.args),
            ));
        }

        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let mut request =
            UpsertPointsBuilder::new(self.args.collection_name.clone(), points.clone())
                .wait(self.args.wait_on_upsert);

        if let Some(ordering) = self.args.write_ordering {
            request = request.ordering(ordering);
        }
        if let Some(shard_key) = &args.shard_key {
            request = request.shard_key_selector(vec![Key::Keyword(shard_key.to_string())]);
        }

        let request = request.build();
        let res = retry_with_clients2(&self.clients, args, |client| {
            client
                .upsert_points(request.clone())
                .map_err(log_points2(points.clone()))
        })
        .await?;

        let latency = res.time;

        self.timings.write().await.push(Timing {
            delay_millis: self.start_time.elapsed().as_millis() as f64,
            value: latency,
        });

        if self.args.set_payload {
            let mut request_builder = SetPayloadPointsBuilder::new(
                self.args.collection_name.clone(),
                random_payload(&self.args),
            )
            .points_selector(batch_ids)
            .wait(self.args.wait_on_upsert);

            if let Some(ordering) = self.args.write_ordering {
                request_builder = request_builder.ordering(ordering);
            }

            let request = request_builder.build();

            retry_with_clients2(&self.clients, args, |client| {
                client.set_payload(request.clone())
            })
            .await?;
        }

        if res.time > self.args.timing_threshold {
            self.progress_bar
                .println(format!("Slow upsert: {:?}", res.time));
        }

        if let Some(delay_millis) = self.args.delay {
            sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        Ok(())
    }

    pub async fn save_data(&self) {
        if let Some(jsonl_path) = &self.args.jsonl_updates {
            save_timings_as_jsonl(
                jsonl_path,
                self.args.absolute_time.unwrap_or(false),
                &self.timings.read().await,
                self.start_timestamp_millis,
                "upsert_latency",
            )
            .unwrap();
        }
    }
}
