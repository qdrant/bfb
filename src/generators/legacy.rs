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
            // Cycle through the file when more points are requested than it holds,
            // like `source: {type: file}` does in YAML configs.
            let n = reader.num_vectors.max(1) as usize;
            reader.read_vector(idx as usize % n).to_vec().into()
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use clap::Parser;
    use qdrant_client::qdrant::vector::Vector as VectorVariant;

    use super::*;

    /// Write `vectors` to a temporary file in fbin layout.
    fn fbin_file(vectors: &[Vec<f32>]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&(vectors.len() as i32).to_le_bytes())
            .unwrap();
        file.write_all(&(vectors[0].len() as i32).to_le_bytes())
            .unwrap();
        for value in vectors.iter().flatten() {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        file
    }

    fn dense(point: PointStruct) -> Vec<f32> {
        match point.vectors.unwrap().vectors_options.unwrap() {
            VectorsOptions::Vector(vector) => match vector.vector.unwrap() {
                VectorVariant::Dense(dense) => dense.data,
                _ => panic!("expected a dense vector"),
            },
            _ => panic!("expected the unnamed default vector"),
        }
    }

    /// `--fbin` with more points (or a larger `--offset`) than the file holds
    /// must cycle through the file instead of panicking on an out-of-range read.
    #[test]
    fn fbin_vectors_wrap_past_the_end_of_the_file() {
        let file = fbin_file(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let reader = FBinReader::new(file.path()).unwrap();
        let args = Args::parse_from(["bfb", "--dim", "2"]);
        let generator = LegacyGenerator::new(args, Some(reader));

        assert_eq!(dense(generator.make_point(0)), [1.0, 2.0]);
        assert_eq!(dense(generator.make_point(1)), [3.0, 4.0]);
        assert_eq!(dense(generator.make_point(2)), [1.0, 2.0]);
        assert_eq!(dense(generator.make_point(5)), [3.0, 4.0]);
    }
}
