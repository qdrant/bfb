//! Point generation, abstracted behind [`PointGenerator`] so the upload
//! pipeline (parallelism, RPS, progress, timings) is shared between the legacy
//! flag-driven path ([`LegacyGenerator`]) and the YAML-config path
//! ([`ConfigGenerator`]).

use std::collections::HashMap;
use std::path::Path;

use qdrant_client::Payload;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{PointId, PointStruct, Vector, Vectors};
use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::seq::IndexedRandom;
use serde_json::json;
use uuid::Uuid;

use crate::args::Args;
use crate::common::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_dense_vector, random_keyword, random_payload,
    random_sparse_vector, random_text, random_vector,
};
use crate::config::{
    DatatypeKind, DistributionKind, FileStrategy, IdType, PayloadSourceKind, PayloadType,
    UploadConfig, VectorConfig, VectorSource,
};
use crate::fbin_reader::FBinReader;

/// Produces the per-point data (id, vectors, payload). The runtime layer owns
/// *which* numeric id to use (offset / max-id logic); the generator owns the
/// *shape*.
pub trait PointGenerator: Send + Sync {
    /// Build the point for the given numeric index.
    fn make_point(&self, idx: u64) -> PointStruct;

    /// Build just a payload (used by the `--set-payload` path).
    fn make_payload(&self) -> Payload;
}

// ------------------------------- Legacy ----------------------------------

/// Reproduces the exact point-generation behaviour of the flag-driven CLI.
pub struct LegacyGenerator {
    args: Args,
    reader: Option<FBinReader>,
    zipf: Option<rand_distr::Zipf<f64>>,
}

impl LegacyGenerator {
    pub fn new(args: Args, reader: Option<FBinReader>) -> Self {
        let zipf = args
            .text_payloads
            .then(|| create_zipf(args.text_payload_vocabulary.unwrap_or(DEFAULT_VOCAB_SIZE)));
        LegacyGenerator { args, reader, zipf }
    }
}

impl PointGenerator for LegacyGenerator {
    fn make_point(&self, idx: u64) -> PointStruct {
        let mut rng = rand::rng();

        let point_id = PointId {
            point_id_options: Some(if self.args.uuids {
                PointIdOptions::Uuid(Uuid::new_v4().to_string())
            } else {
                PointIdOptions::Num(idx)
            }),
        };

        let vectors: Vectors = if let Some(reader) = &self.reader {
            reader.read_vector(idx as usize).to_vec().into()
        } else if self.args.vectors_per_point != 1 {
            let vectors_map: HashMap<_, _> = (0..self.args.vectors_per_point)
                .map(|i| (format!("{i}"), random_vector(&mut rng, &self.args)))
                .collect();
            vectors_map.into()
        } else {
            random_vector(&mut rng, &self.args).into()
        };

        let vectors: Vectors = if let Some(sparsity) = self.args.sparse_vectors {
            let mut vectors_map: HashMap<_, _> = Default::default();

            for i in 0..self.args.sparse_vectors_per_point {
                let vector_name = format!("{i}_sparse");
                let vector = Vector::from(random_sparse_vector(
                    &mut rng,
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

        PointStruct::new(
            point_id,
            vectors,
            random_payload(&mut rng, &self.args, self.zipf.as_ref()),
        )
    }

    fn make_payload(&self) -> Payload {
        let mut rng = rand::rng();
        random_payload(&mut rng, &self.args, self.zipf.as_ref())
    }
}

// ------------------------------- Config ----------------------------------

/// Generates points from a parsed YAML [`UploadConfig`].
pub struct ConfigGenerator {
    config: UploadConfig,
    /// File readers aligned with `config.collection.vectors` (`None` = random).
    readers: Vec<Option<FBinReader>>,
    /// Zipf distributions aligned with `config.collection.payloads` (text only).
    text_zipf: Vec<Option<rand_distr::Zipf<f64>>>,
    /// Cluster centers aligned with `config.collection.payloads` (geo clusters).
    geo_clusters: Vec<Option<Vec<(f64, f64)>>>,
    /// Zipf distributions aligned with `config.collection.sparse_vectors`.
    sparse_zipf: Vec<Option<rand_distr::Zipf<f64>>>,
}

/// Uniformly-sampled bag-of-words text (the non-zipf counterpart to
/// [`random_text`]).
fn uniform_text(rng: &mut impl Rng, num_words: usize, vocab: usize) -> String {
    let vocab = vocab.max(1);
    (0..num_words)
        .map(|_| format!("word_{}", rng.random_range(0..vocab)))
        .collect::<Vec<_>>()
        .join(" ")
}

const GEO_CENTER_LAT: f64 = 52.52437;
const GEO_CENTER_LON: f64 = 13.41053;
const GEO_SPREAD_DEG: f64 = 1.0;
const GEO_CLUSTER_JITTER_DEG: f64 = 0.02;

impl ConfigGenerator {
    pub fn new(config: &UploadConfig) -> anyhow::Result<Self> {
        let mut rng = rand::rng();

        let readers = config
            .collection
            .vectors
            .iter()
            .map(|v| match &v.source {
                VectorSource::File { path, .. } => Some(FBinReader::new(Path::new(path))),
                VectorSource::Random => None,
            })
            .collect();

        let text_zipf = config
            .collection
            .payloads
            .iter()
            .map(|p| {
                (p.kind == PayloadType::Text && p.source.distribution == DistributionKind::Zipf)
                    .then(|| create_zipf(p.source.vocab_size.unwrap_or(DEFAULT_VOCAB_SIZE)))
            })
            .collect();

        let geo_clusters = config
            .collection
            .payloads
            .iter()
            .map(|p| {
                (p.kind == PayloadType::Geo && p.source.kind == PayloadSourceKind::RandomClusters)
                    .then(|| {
                        let count = p.source.clusters.unwrap_or(10);
                        (0..count)
                            .map(|_| {
                                (
                                    GEO_CENTER_LAT
                                        + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                                    GEO_CENTER_LON
                                        + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                                )
                            })
                            .collect()
                    })
            })
            .collect();

        let sparse_zipf = config
            .collection
            .sparse_vectors
            .iter()
            .map(|s| {
                (s.source.distribution == DistributionKind::Zipf).then(|| create_zipf(s.source.dim))
            })
            .collect();

        Ok(ConfigGenerator {
            config: config.clone(),
            readers,
            text_zipf,
            geo_clusters,
            sparse_zipf,
        })
    }

    fn gen_one_vector(
        &self,
        vc: &VectorConfig,
        reader: &Option<FBinReader>,
        idx: u64,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        match (&vc.source, reader) {
            (VectorSource::File { strategy, .. }, Some(reader)) => {
                let n = reader.num_vectors.max(1) as usize;
                let file_idx = match strategy {
                    FileStrategy::FromStart => (idx as usize) % n,
                    FileStrategy::RandomSample => rng.random_range(0..n),
                };
                reader.read_vector(file_idx).to_vec()
            }
            _ => {
                let is_uint = vc.datatype == DatatypeKind::Uint8;
                random_dense_vector(rng, vc.size as usize, is_uint)
            }
        }
    }

    fn gen_dense(&self, i: usize, vc: &VectorConfig, idx: u64, rng: &mut impl Rng) -> Vector {
        let reader = &self.readers[i];
        if let Some(mv) = &vc.multivector {
            let multi: Vec<_> = (0..mv.count)
                .map(|_| self.gen_one_vector(vc, reader, idx, rng))
                .collect();
            Vector::new_multi(multi)
        } else {
            self.gen_one_vector(vc, reader, idx, rng).into()
        }
    }

    fn gen_sparse(&self, i: usize, rng: &mut impl Rng) -> Vector {
        let sc = &self.config.collection.sparse_vectors[i];
        match &self.sparse_zipf[i] {
            Some(zipf) => {
                // Zipf-distributed indices: sample ~dim*sparsity dims, dedup.
                let target = ((sc.source.dim as f64) * sc.source.sparsity).ceil() as usize;
                let mut seen = std::collections::HashSet::new();
                let mut pairs = Vec::with_capacity(target);
                let mut attempts = 0;
                while pairs.len() < target && attempts < target * 8 {
                    attempts += 1;
                    let dim = (zipf.sample(rng) as u32).max(1);
                    if seen.insert(dim) {
                        pairs.push((dim, rng.random_range(0.0..10.0) as f32));
                    }
                }
                Vector::from(pairs)
            }
            None => Vector::from(random_sparse_vector(rng, sc.source.dim, sc.source.sparsity)),
        }
    }

    fn gen_payload(&self, rng: &mut impl Rng) -> Payload {
        let mut payload = Payload::with_capacity(self.config.collection.payloads.len());

        for (i, pc) in self.config.collection.payloads.iter().enumerate() {
            let src = &pc.source;
            match pc.kind {
                PayloadType::Keyword => {
                    let card = src.cardinality.unwrap_or(100);
                    let mult = src.length_multiplier.unwrap_or(1);
                    let vpp = src.values_per_point.unwrap_or(1);
                    if vpp <= 1 {
                        payload.insert(pc.name.clone(), random_keyword(rng, card, mult));
                    } else {
                        let count = rng.random_range(1..=vpp);
                        let values: Vec<_> = (0..count)
                            .map(|_| random_keyword(rng, card, mult))
                            .collect();
                        payload.insert(pc.name.clone(), values);
                    }
                }
                PayloadType::Integer => {
                    let min = src.min.unwrap_or(0.0) as i64;
                    let max = src.max.unwrap_or(100.0) as i64;
                    let max = max.max(min + 1);
                    let vpp = src.values_per_point.unwrap_or(1);
                    if vpp <= 1 {
                        payload.insert(pc.name.clone(), rng.random_range(min..max));
                    } else {
                        let count = rng.random_range(1..=vpp);
                        let values: Vec<i64> =
                            (0..count).map(|_| rng.random_range(min..max)).collect();
                        payload.insert(pc.name.clone(), values);
                    }
                }
                PayloadType::Float => {
                    let min = src.min.unwrap_or(-1.0);
                    let max = src
                        .max
                        .unwrap_or(1.0)
                        .max(src.min.unwrap_or(-1.0) + f64::EPSILON);
                    payload.insert(pc.name.clone(), rng.random_range(min..max));
                }
                PayloadType::Bool => {
                    payload.insert(
                        pc.name.clone(),
                        rng.random_bool(src.true_ratio.unwrap_or(0.5)),
                    );
                }
                PayloadType::Uuid => {
                    payload.insert(pc.name.clone(), Uuid::new_v4().to_string());
                }
                PayloadType::Geo => {
                    let (lat, lon) = match &self.geo_clusters[i] {
                        Some(centers) => {
                            let &(clat, clon) = centers.choose(rng).unwrap();
                            (
                                clat + rng
                                    .random_range(-GEO_CLUSTER_JITTER_DEG..GEO_CLUSTER_JITTER_DEG),
                                clon + rng
                                    .random_range(-GEO_CLUSTER_JITTER_DEG..GEO_CLUSTER_JITTER_DEG),
                            )
                        }
                        None => (
                            GEO_CENTER_LAT + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                            GEO_CENTER_LON + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                        ),
                    };
                    payload.insert(pc.name.clone(), json!({ "lat": lat, "lon": lon }));
                }
                PayloadType::Text => {
                    let min_len = src.min_length.unwrap_or(16);
                    let max_len = src.max_length.unwrap_or(min_len).max(min_len);
                    let len = if max_len > min_len {
                        rng.random_range(min_len..=max_len)
                    } else {
                        min_len
                    };
                    let text = match &self.text_zipf[i] {
                        Some(zipf) => random_text(rng, len, zipf),
                        None => {
                            uniform_text(rng, len, src.vocab_size.unwrap_or(DEFAULT_VOCAB_SIZE))
                        }
                    };
                    payload.insert(pc.name.clone(), text);
                }
                PayloadType::Datetime => {
                    let value = match src.kind {
                        PayloadSourceKind::Now => chrono::Utc::now(),
                        _ => {
                            // Random within the last year.
                            let secs = rng.random_range(0..365 * 24 * 3600i64);
                            chrono::Utc::now() - chrono::Duration::seconds(secs)
                        }
                    };
                    payload.insert(pc.name.clone(), value.to_rfc3339());
                }
            }
        }

        payload
    }
}

impl PointGenerator for ConfigGenerator {
    fn make_point(&self, idx: u64) -> PointStruct {
        let mut rng = rand::rng();
        let collection = &self.config.collection;

        let point_id = PointId {
            point_id_options: Some(match collection.id {
                IdType::Integer => PointIdOptions::Num(idx),
                IdType::Uuid => PointIdOptions::Uuid(Uuid::new_v4().to_string()),
            }),
        };

        let mut named: HashMap<String, Vector> = HashMap::new();
        let mut unnamed: Option<Vector> = None;

        for (i, vc) in collection.vectors.iter().enumerate() {
            let vector = self.gen_dense(i, vc, idx, &mut rng);
            match &vc.name {
                Some(name) => {
                    named.insert(name.clone(), vector);
                }
                None => unnamed = Some(vector),
            }
        }

        for (i, sc) in collection.sparse_vectors.iter().enumerate() {
            named.insert(sc.name.clone(), self.gen_sparse(i, &mut rng));
        }

        let vectors: Vectors = if named.is_empty() {
            unnamed.expect("validated: at least one vector").into()
        } else {
            if let Some(unnamed) = unnamed {
                named.insert("".to_string(), unnamed);
            }
            named.into()
        };

        PointStruct::new(point_id, vectors, self.gen_payload(&mut rng))
    }

    fn make_payload(&self) -> Payload {
        let mut rng = rand::rng();
        self.gen_payload(&mut rng)
    }
}
