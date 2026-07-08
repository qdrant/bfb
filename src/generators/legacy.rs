//! Flag-driven point generation, reproducing the exact behaviour of the
//! legacy CLI (`bfb` without a YAML config).

use std::collections::HashMap;

use qdrant_client::Payload;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{PointId, PointStruct, Vector, Vectors};
use uuid::Uuid;

use super::PointGenerator;
use super::random::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_payload, random_sparse_vector, random_vector,
};
use crate::args::Args;
use crate::fbin_reader::FBinReader;

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
                let vocab_size = self.args.sparse_dim.unwrap_or(self.args.dim);
                let length = ((vocab_size as f64) * sparsity).ceil() as usize;
                let vector = Vector::from(random_sparse_vector(&mut rng, vocab_size, length));
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
