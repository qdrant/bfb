//! YAML-config-driven point generation ([`ConfigGenerator`]).

use std::collections::HashMap;

use qdrant_client::Payload;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{PointId, PointStruct, Vector, Vectors};
use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::seq::IndexedRandom;
use serde_json::json;
use uuid::Uuid;

use super::PointGenerator;
use super::random::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_dense_vector, random_keyword, random_sparse_vector,
    random_text,
};
use crate::config::{
    DatatypeKind, DistributionKind, FileStrategy, IdType, PayloadSource, PayloadSourceKind,
    PayloadType, UploadConfig, VectorConfig, VectorSource,
};
use crate::dataset::{UploadDatasetSources, default_datasets_dir, ensure_local_file};
use crate::fbin_reader::FBinReader;

/// Generates points from a parsed YAML [`UploadConfig`].
pub struct ConfigGenerator {
    config: UploadConfig,
    /// File readers aligned with `config.collection.vectors` (`None` = random).
    readers: Vec<Option<FBinReader>>,
    /// Dataset readers for vectors, sparse vectors, and payloads.
    datasets: UploadDatasetSources,
    /// Zipf distributions aligned with `config.collection.fields` (text only).
    text_zipf: Vec<Option<rand_distr::Zipf<f64>>>,
    /// Cluster centers aligned with `config.collection.fields` (geo clusters).
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
        Self::new_with_datasets_dir(config, &default_datasets_dir())
    }

    pub fn new_with_datasets_dir(
        config: &UploadConfig,
        datasets_dir: &std::path::Path,
    ) -> anyhow::Result<Self> {
        let mut rng = rand::rng();

        let readers = config
            .collection
            .vectors
            .iter()
            .map(|v| match &v.source {
                VectorSource::File { path, .. } => {
                    let local = ensure_local_file(datasets_dir, path)?;
                    FBinReader::new(&local).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let datasets = UploadDatasetSources::open(config, datasets_dir)?;

        let text_zipf = config
            .collection
            .fields
            .iter()
            .map(|p| {
                let src = p.source.as_ref();
                (p.kind == PayloadType::Text
                    && src.map(|s| s.distribution) == Some(DistributionKind::Zipf))
                .then(|| create_zipf(src.and_then(|s| s.vocab_size).unwrap_or(DEFAULT_VOCAB_SIZE)))
            })
            .collect();

        let geo_clusters = config
            .collection
            .fields
            .iter()
            .map(|p| {
                let src = p.source.as_ref();
                (p.kind == PayloadType::Geo
                    && src.map(|s| s.kind) == Some(PayloadSourceKind::RandomClusters))
                .then(|| {
                    let count = src.and_then(|s| s.clusters).unwrap_or(10);
                    (0..count)
                        .map(|_| {
                            (
                                GEO_CENTER_LAT + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                                GEO_CENTER_LON + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
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
                (s.source.distribution == DistributionKind::Zipf)
                    .then(|| create_zipf(s.source.vocab_size))
            })
            .collect();

        Ok(ConfigGenerator {
            config: config.clone(),
            readers,
            datasets,
            text_zipf,
            geo_clusters,
            sparse_zipf,
        })
    }

    fn gen_one_vector(
        &self,
        vc: &VectorConfig,
        reader: &Option<FBinReader>,
        slot: usize,
        idx: u64,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        if let VectorSource::Dataset { .. } = &vc.source
            && let Some(vector) = self.datasets.dense_vector(slot, idx)
        {
            return vector;
        }
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
            if let VectorSource::Dataset { .. } = &vc.source
                && let Some(multi) = self.datasets.multi_dense_vector(i, idx)
            {
                return Vector::new_multi(multi);
            }
            let multi: Vec<_> = (0..mv.count)
                .map(|_| self.gen_one_vector(vc, reader, i, idx, rng))
                .collect();
            Vector::new_multi(multi)
        } else {
            self.gen_one_vector(vc, reader, i, idx, rng).into()
        }
    }

    fn gen_sparse(&self, i: usize, idx: u64, rng: &mut impl Rng) -> Vector {
        let sc = &self.config.collection.sparse_vectors[i];
        if let Some(pairs) = self.datasets.sparse_vector(i, idx) {
            return Vector::from(pairs);
        }
        match &self.sparse_zipf[i] {
            Some(zipf) => {
                // Zipf-distributed indices: sample `length` dims, dedup.
                let target = sc.source.length;
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
            None => Vector::from(random_sparse_vector(
                rng,
                sc.source.vocab_size,
                sc.source.length,
            )),
        }
    }

    fn gen_payload(&self, idx: u64, rng: &mut impl Rng) -> Payload {
        let mut payload = Payload::with_capacity(self.config.collection.fields.len());

        // Collection-level whole-payload source: insert every field of the point's
        // payload object. Per-field sources below may override individual keys.
        let has_object = if let Some(fields) = self.datasets.payload_object(idx) {
            for (key, value) in fields {
                payload.insert(key, value);
            }
            true
        } else {
            false
        };

        let default_src = PayloadSource::default();
        for (i, pc) in self.config.collection.fields.iter().enumerate() {
            if let Some(value) = self.datasets.payload_value(i, idx) {
                payload.insert(pc.name.clone(), value);
                continue;
            }

            let src = match &pc.source {
                Some(src) => src,
                // Index-only field: its value comes from the whole-payload object.
                None if has_object => continue,
                // No source and no object: fall back to random generation.
                None => &default_src,
            };
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
            named.insert(sc.name.clone(), self.gen_sparse(i, idx, &mut rng));
        }

        let vectors: Vectors = if named.is_empty() {
            unnamed.expect("validated: at least one vector").into()
        } else {
            if let Some(unnamed) = unnamed {
                named.insert("".to_string(), unnamed);
            }
            named.into()
        };

        PointStruct::new(point_id, vectors, self.gen_payload(idx, &mut rng))
    }

    fn make_payload(&self) -> Payload {
        let mut rng = rand::rng();
        self.gen_payload(0, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::vectors::VectorsOptions;
    use std::collections::HashMap;

    fn build_gen(yaml: &str) -> ConfigGenerator {
        let config: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        ConfigGenerator::new(&config).unwrap()
    }

    /// Serialize `vectors` in fbin layout: `[i32 count][i32 dim][f32 data…]`.
    fn fbin_bytes(vectors: &[Vec<f32>]) -> Vec<u8> {
        let dim = vectors[0].len();
        let mut bytes = (vectors.len() as i32).to_le_bytes().to_vec();
        bytes.extend((dim as i32).to_le_bytes());
        for v in vectors {
            for f in v {
                bytes.extend(f.to_le_bytes());
            }
        }
        bytes
    }

    /// A `source: {type: file}` pointing at an http(s) URL must download the
    /// file and generate points from it, not panic opening a path named "http:".
    #[test]
    fn generates_vectors_from_a_remote_fbin_file() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let (url, server) = crate::dataset::test_http::serve_once(fbin_bytes(&vectors), 1);
        let dir = tempfile::tempdir().unwrap();

        let yaml = format!(
            r#"
collection:
  name: t
  vectors:
    - size: 4
      source:
        type: file
        path: {url}
        strategy: from-start
"#
        );
        let config: UploadConfig = serde_yaml::from_str(&yaml).unwrap();
        config.validate().unwrap();

        let generator = ConfigGenerator::new_with_datasets_dir(&config, dir.path()).unwrap();
        let point = generator.make_point(0);

        let data = match point.vectors.unwrap().vectors_options.unwrap() {
            VectorsOptions::Vector(v) => match v.vector.unwrap() {
                qdrant_client::qdrant::vector::Vector::Dense(d) => d.data,
                _ => panic!("expected a dense vector"),
            },
            _ => panic!("expected the unnamed default vector"),
        };
        assert_eq!(data, vectors[0], "point 0 must come from the remote file");
        assert_eq!(server.join().unwrap(), 1);
    }

    /// The LAION shape: dense vectors from a bare `.npy`, the whole payload
    /// object from a parquet of the same row count. Both are indexed by point
    /// id, so row *i* of each file must land on point *i*.
    #[test]
    fn pairs_npy_vectors_with_parquet_payloads() {
        use crate::dataset::fixtures::{make_ramp_npy, write_parquet};

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("emb.npy"), make_ramp_npy(0, 8, 4)).unwrap();
        write_parquet(&dir.path().join("meta.parquet"), 0, 8, 4);

        let config: UploadConfig = serde_yaml::from_str(
            "
collection:
  name: t
  vectors:
    - size: 4
      source:
        type: dataset
        name: emb
        format: npy
        path: emb.npy
  payload:
    source:
      type: dataset
      dataset:
        name: meta
        format: parquet
        path: meta.parquet
        exclude: [url]
  fields:
    - name: similarity
      type: float
",
        )
        .unwrap();
        config.validate().unwrap();

        let generator = ConfigGenerator::new_with_datasets_dir(&config, dir.path()).unwrap();

        let point = generator.make_point(5);
        let data = match point.vectors.unwrap().vectors_options.unwrap() {
            VectorsOptions::Vector(v) => match v.vector.unwrap() {
                qdrant_client::qdrant::vector::Vector::Dense(d) => d.data,
                _ => panic!("expected a dense vector"),
            },
            _ => panic!("expected the unnamed default vector"),
        };
        assert_eq!(data, vec![20.0, 21.0, 22.0, 23.0], "row 5 of the .npy");

        let payload: serde_json::Value =
            serde_json::to_value(point.payload.into_iter().collect::<HashMap<_, _>>()).unwrap();
        assert_eq!(payload["id"], 5, "row 5 of the parquet");
        assert_eq!(payload["similarity"], 5.0);
        assert_eq!(payload["caption"], "caption 5");
        assert!(
            payload.get("url").is_none(),
            "`exclude` must drop the column: {payload}"
        );
    }

    fn named(point: &PointStruct) -> HashMap<String, Vector> {
        match point.vectors.clone().unwrap().vectors_options.unwrap() {
            VectorsOptions::Vectors(nv) => nv.vectors,
            _ => panic!("expected named vectors"),
        }
    }

    fn dense_dim(v: &Vector) -> usize {
        match v.vector.as_ref().unwrap() {
            qdrant_client::qdrant::vector::Vector::Dense(d) => d.data.len(),
            _ => panic!("expected dense vector"),
        }
    }

    #[test]
    fn point_id_integer_and_uuid() {
        let g = build_gen("collection:\n  id: integer\n  vectors:\n    - size: 8\n");
        assert!(matches!(
            g.make_point(7).id.unwrap().point_id_options,
            Some(PointIdOptions::Num(7))
        ));

        let g = build_gen("collection:\n  id: uuid\n  vectors:\n    - size: 8\n");
        assert!(matches!(
            g.make_point(7).id.unwrap().point_id_options,
            Some(PointIdOptions::Uuid(_))
        ));
    }

    #[test]
    fn unnamed_vector_is_single() {
        let g = build_gen("collection:\n  vectors:\n    - size: 8\n");
        assert!(matches!(
            g.make_point(0).vectors.unwrap().vectors_options.unwrap(),
            VectorsOptions::Vector(_)
        ));
    }

    #[test]
    fn named_vectors_sparse_and_dims() {
        let g = build_gen(
            "collection:
  vectors:
    - name: image
      size: 16
    - name: text
      size: 8
  sparse_vectors:
    - name: bm25
      source: { type: random, vocab_size: 100, length: 20 }
",
        );
        let point = g.make_point(1);
        let nv = named(&point);

        let mut keys: Vec<_> = nv.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["bm25", "image", "text"]);

        assert_eq!(dense_dim(&nv["image"]), 16);
        assert_eq!(dense_dim(&nv["text"]), 8);
        assert!(matches!(
            nv["bm25"].vector.as_ref().unwrap(),
            qdrant_client::qdrant::vector::Vector::Sparse(_)
        ));
    }

    #[test]
    fn multivector_arity() {
        let g = build_gen(
            "collection:\n  vectors:\n    - name: m\n      size: 4\n      multivector: { count: 3 }\n",
        );
        let point = g.make_point(0);
        match named(&point)["m"].vector.as_ref().unwrap() {
            qdrant_client::qdrant::vector::Vector::MultiDense(m) => {
                assert_eq!(m.vectors.len(), 3);
                assert!(m.vectors.iter().all(|v| v.data.len() == 4));
            }
            _ => panic!("expected multidense vector"),
        }
    }

    /// A `format: multivector` dataset source must read each point's real,
    /// ragged sub-vectors from `vectors.npy`/`offsets.npy`, not repeat one row
    /// `multivector.count` times.
    #[test]
    fn reads_multivectors_from_a_dataset() {
        use crate::dataset::fixtures::make_ramp_npy;

        let dir = tempfile::tempdir().unwrap();
        let mv_dir = dir.path().join("colbert");
        std::fs::create_dir(&mv_dir).unwrap();
        // 5 sub-vectors total, dim 4: point 0 -> rows [0,2), point 1 -> [2,5).
        std::fs::write(mv_dir.join("vectors.npy"), make_ramp_npy(0, 5, 4)).unwrap();
        std::fs::write(mv_dir.join("offsets.npy"), make_offsets_npy(&[0i64, 2, 5])).unwrap();

        let config: UploadConfig = serde_yaml::from_str(
            "
collection:
  name: t
  vectors:
    - name: m
      size: 4
      multivector: { count: 1 }
      source:
        type: dataset
        name: colbert
        format: multivector
        path: colbert
",
        )
        .unwrap();
        config.validate().unwrap();

        let generator = ConfigGenerator::new_with_datasets_dir(&config, dir.path()).unwrap();

        let point0 = generator.make_point(0);
        match named(&point0)["m"].vector.as_ref().unwrap() {
            qdrant_client::qdrant::vector::Vector::MultiDense(m) => {
                assert_eq!(m.vectors.len(), 2, "point 0 has 2 sub-vectors, not `count`");
                assert_eq!(m.vectors[0].data, vec![0.0, 1.0, 2.0, 3.0]);
                assert_eq!(m.vectors[1].data, vec![4.0, 5.0, 6.0, 7.0]);
            }
            _ => panic!("expected multidense vector"),
        }

        let point1 = generator.make_point(1);
        match named(&point1)["m"].vector.as_ref().unwrap() {
            qdrant_client::qdrant::vector::Vector::MultiDense(m) => {
                assert_eq!(m.vectors.len(), 3, "point 1 has 3 sub-vectors");
            }
            _ => panic!("expected multidense vector"),
        }
    }

    fn make_offsets_npy(offsets: &[i64]) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}",
            offsets.len()
        );
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(header.len() as u16).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        for &o in offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        buf
    }

    #[test]
    fn payload_fields_and_types() {
        let g = build_gen(
            "collection:
  vectors:
    - size: 4
  fields:
    - name: color
      type: keyword
      source: { type: random, cardinality: 10 }
    - name: price
      type: integer
      source: { type: random, min: 5, max: 9 }
    - name: ok
      type: bool
    - name: loc
      type: geo
",
        );
        let value = g.make_payload().deserialize::<serde_json::Value>().unwrap();
        let obj = value.as_object().unwrap();

        assert!(obj["color"].is_string());
        let price = obj["price"].as_i64().unwrap();
        assert!((5..9).contains(&price), "price {price} out of [5, 9)");
        assert!(obj["ok"].is_boolean());
        let loc = obj["loc"].as_object().unwrap();
        assert!(loc.contains_key("lat") && loc.contains_key("lon"));
    }

    #[test]
    fn keyword_uses_configured_cardinality() {
        // 100 points × low cardinality ⇒ all values fall in keyword_0..keyword_5.
        let g = build_gen(
            "collection:\n  vectors:\n    - size: 4\n  fields:\n    - name: c\n      type: keyword\n      source: { type: random, cardinality: 5 }\n",
        );
        for _ in 0..100 {
            let value = g.make_payload().deserialize::<serde_json::Value>().unwrap();
            let kw = value["c"].as_str().unwrap().to_string();
            let n: usize = kw.strip_prefix("keyword_").unwrap().parse().unwrap();
            assert!(n < 5, "keyword index {n} exceeds cardinality");
        }
    }
}
