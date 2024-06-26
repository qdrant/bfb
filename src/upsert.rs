use crate::common::{random_sparse_vector, random_vector, retry_with_clients};
use crate::fbin_reader::FBinReader;
use crate::{random_dense_vector, random_payload, Args};
use anyhow::Error;
use indicatif::ProgressBar;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{PointId, PointStruct, PointsSelector, Vector, Vectors};
use rand::Rng;
use std::cmp::min;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::sleep;

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
                random_payload(
                    self.args.keywords,
                    self.args.float_payloads,
                    self.args.int_payloads,
                ),
            ));
        }

        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let ordering = self.args.write_ordering.map(Into::into);

        let res = if self.args.wait_on_upsert {
            retry_with_clients(&self.clients, args, |client| {
                client.upsert_points_blocking(
                    &self.args.collection_name,
                    None,
                    points.clone(),
                    ordering.clone(),
                )
            })
            .await?
        } else {
            retry_with_clients(&self.clients, args, |client| {
                client.upsert_points(
                    &self.args.collection_name,
                    None,
                    points.clone(),
                    ordering.clone(),
                )
            })
            .await?
        };

        if self.args.set_payload {
            let points: PointsSelector = batch_ids.into();
            if self.args.wait_on_upsert {
                retry_with_clients(&self.clients, args, |client| {
                    client.set_payload_blocking(
                        &self.args.collection_name,
                        None,
                        &points,
                        random_payload(
                            self.args.keywords,
                            self.args.float_payloads,
                            self.args.int_payloads,
                        ),
                        None,
                        ordering.clone(),
                    )
                })
                .await?;
            } else {
                retry_with_clients(&self.clients, args, |client| {
                    client.set_payload(
                        &self.args.collection_name,
                        None,
                        &points,
                        random_payload(
                            self.args.keywords,
                            self.args.float_payloads,
                            self.args.int_payloads,
                        ),
                        None,
                        ordering.clone(),
                    )
                })
                .await?;
            }
        }

        if res.time > self.args.timing_threshold {
            self.progress_bar
                .println(format!("Slow upsert: {:?}", res.time));
        }

        if let Some(delay_millis) = self.args.delay {
            sleep(std::time::Duration::from_millis(delay_millis as u64)).await;
        }

        Ok::<(), Error>(())
    }
}
