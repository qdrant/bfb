use std::cmp::min;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
use rand::RngExt;
use tokio::time::sleep;

use crate::args::Args;
use crate::common::{
    DEFAULT_VOCAB_SIZE, Timing, create_zipf, random_payload, random_sparse_vector, random_vector,
    retry_with_clients,
};
use crate::fbin_reader::FBinReader;
use crate::save_jsonl::save_timings_as_jsonl;

fn log_points(points: &[PointStruct]) -> impl FnOnce(QdrantError) -> QdrantError + use<'_> {
    move |e| {
        let mut point_ids = Vec::with_capacity(points.len());

        for p in points {
            if let Some(point_id_option) = p.id.clone().unwrap().point_id_options {
                match point_id_option {
                    PointIdOptions::Num(num) => point_ids.push(num.to_string()),
                    PointIdOptions::Uuid(uuid) => point_ids.push(uuid.to_string()),
                }
            }
        }
        tracing::warn!(
            "Failed while upserting. point_ids={:?} error={e:?}",
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
    timings: Mutex<Vec<Timing>>,
    zipf: Option<rand_distr::Zipf<f64>>,
}

impl UpsertProcessor {
    pub fn new(
        args: Args,
        stopped: Arc<AtomicBool>,
        clients: Vec<Qdrant>,
        progress_bar: Arc<ProgressBar>,
        reader: Option<FBinReader>,
    ) -> Self {
        let zipf = args
            .text_payloads
            .then(|| create_zipf(args.text_payload_vocabulary.unwrap_or(DEFAULT_VOCAB_SIZE)));

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
            timings: Mutex::new(Vec::new()),
            zipf,
        }
    }

    pub async fn upsert(&self, batch_id: usize, args: &Args) -> Result<(), Error> {
        let points_uploaded = self.args.batch_size * batch_id;
        let points_left = self.args.num_vectors.saturating_sub(points_uploaded);

        if points_left == 0 {
            return Ok(());
        }

        let mut rng = rand::rng();

        let batch_size = min(self.args.batch_size, points_left);
        let mut points = Vec::with_capacity(batch_size);
        let mut batch_ids = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let idx = if let Some(max_id) = self.args.max_id {
                rng.random_range(self.args.offset..max_id) as u64
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
                        let vector_name = format!("{i}");
                        let vector = random_vector(&mut rng, &self.args);
                        (vector_name, vector)
                    })
                    .collect();
                vectors_map.into()
            } else {
                random_vector(&mut rng, &self.args).into()
            };

            let vectors: Vectors = if self.args.use_sparse_vectors() {
                let mut vectors_map: HashMap<_, _> = Default::default();

                for i in 0..self.args.sparse_vectors_per_point {
                    let vector_name = format!("{i}_sparse");
                    let vector = Vector::from(random_sparse_vector(
                        &mut rng,
                        self.args.sparse_vocab_size(),
                        self.args.sparse_avg_dim(),
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
                random_payload(&mut rng, &self.args, self.zipf.as_ref()),
            ));
        }

        if self.stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        let mut request = UpsertPointsBuilder::new(self.args.collection_name.clone(), points)
            .wait(self.args.wait_on_upsert);

        if let Some(ordering) = self.args.write_ordering {
            request = request.ordering(ordering);
        }
        if let Some(shard_key) = &args.shard_key {
            request = request.shard_key_selector(vec![Key::Keyword(shard_key.to_string())]);
        }
        if let Some(timeout) = self.args.timeout {
            request = request.timeout(timeout as u64);
        }

        let request = request.build();
        let res = retry_with_clients(&self.clients, args, |client| {
            client
                .upsert_points(request.clone())
                .map_err(log_points(&request.points))
        })
        .await?;

        let latency = res.time;

        self.timings.lock().unwrap().push(Timing {
            delay_millis: self.start_time.elapsed().as_millis() as u32,
            value: latency as f32,
        });

        if self.args.set_payload {
            let mut request_builder = SetPayloadPointsBuilder::new(
                self.args.collection_name.clone(),
                random_payload(&mut rng, &self.args, self.zipf.as_ref()),
            )
            .points_selector(batch_ids)
            .wait(self.args.wait_on_upsert);

            if let Some(ordering) = self.args.write_ordering {
                request_builder = request_builder.ordering(ordering);
            }
            if let Some(timeout) = self.args.timeout {
                request_builder = request_builder.timeout(timeout as u64);
            }

            let request = request_builder.build();

            retry_with_clients(&self.clients, args, |client| {
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

    pub fn save_data(&self) {
        if let Some(jsonl_path) = &self.args.jsonl_updates {
            save_timings_as_jsonl(
                jsonl_path,
                self.args.absolute_time.unwrap_or(false),
                &self.timings.lock().unwrap(),
                self.start_timestamp_millis,
                "upsert_latency",
            )
            .unwrap();
        }
    }
}
