use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use super::config::{DatasetConfig, DatasetKind};
use super::download::ensure_downloaded;
use super::parts::PartitionedReader;
use super::readers::{
    H5Reader, MultivectorReader, NpyReader, ParquetReader, QueryEntry, SparseReader, SparseVector,
    TarReader,
};
use super::registry::load_registry;

enum DatasetReaderInner {
    H5(H5Reader),
    Tar(TarReader),
    Sparse(SparseReader),
    Npy(NpyReader),
    Parquet(ParquetReader),
    Multivector(MultivectorReader),
    /// A `parts:` family read as one row space; the part format is `npy` or
    /// `parquet`, so it answers the same accessors as those two.
    Partitioned(PartitionedReader),
}

/// Random access to points from a vector-db-benchmark dataset.
pub struct DatasetReader {
    inner: DatasetReaderInner,
    pub num_points: usize,
}

impl DatasetReader {
    pub fn open(datasets_dir: &Path, config: &DatasetConfig) -> Result<Self> {
        let registry = load_registry(datasets_dir)?;
        let config = DatasetConfig::resolve(config.clone(), &registry)?;

        if config.parts.is_some() {
            let reader = PartitionedReader::open(datasets_dir, &config)?;
            let n = reader.num_points();
            return Ok(DatasetReader {
                inner: DatasetReaderInner::Partitioned(reader),
                num_points: n,
            });
        }

        let local_path = ensure_downloaded(datasets_dir, &config)?;
        let (inner, num_points) = match config.kind {
            DatasetKind::H5 => {
                let reader = H5Reader::open(&local_path)?;
                let n = reader.num_points();
                (DatasetReaderInner::H5(reader), n)
            }
            DatasetKind::Tar => {
                let reader = TarReader::open(&local_path)?;
                let n = reader.num_points();
                (DatasetReaderInner::Tar(reader), n)
            }
            DatasetKind::Sparse => {
                let reader = SparseReader::open(&local_path)?;
                let n = reader.num_points();
                (DatasetReaderInner::Sparse(reader), n)
            }
            DatasetKind::Npy => {
                let reader = NpyReader::open(&local_path)?;
                let n = reader.num_points();
                (DatasetReaderInner::Npy(reader), n)
            }
            DatasetKind::Parquet => {
                let reader = ParquetReader::open(
                    &local_path,
                    config.columns.as_deref(),
                    &config.exclude,
                    config.fill_null.as_ref(),
                )?;
                let n = reader.num_points();
                (DatasetReaderInner::Parquet(reader), n)
            }
            DatasetKind::Multivector => {
                let reader = MultivectorReader::open(&local_path)?;
                let n = reader.num_points();
                (DatasetReaderInner::Multivector(reader), n)
            }
        };
        Ok(DatasetReader { inner, num_points })
    }

    pub fn dense_vector(&self, idx: usize) -> Result<Vec<f32>> {
        match &self.inner {
            DatasetReaderInner::H5(r) => r.vector_at(idx),
            DatasetReaderInner::Tar(r) => r.vector_at(idx),
            DatasetReaderInner::Npy(r) => r.vector_at(idx),
            DatasetReaderInner::Partitioned(r) => r.vector_at(idx),
            DatasetReaderInner::Sparse(_)
            | DatasetReaderInner::Parquet(_)
            | DatasetReaderInner::Multivector(_) => {
                bail!("dataset does not contain dense vectors")
            }
        }
    }

    pub fn sparse_vector(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        match &self.inner {
            DatasetReaderInner::Sparse(r) => r.vector_at(idx),
            _ => bail!("dataset does not contain sparse vectors"),
        }
    }

    /// A point's sub-vectors from a `multivector` dataset (ColBERT-style).
    pub fn multi_dense_vector(&self, idx: usize) -> Result<Vec<Vec<f32>>> {
        match &self.inner {
            DatasetReaderInner::Multivector(r) => r.vector_at(idx),
            _ => bail!("dataset does not contain multivectors"),
        }
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        match &self.inner {
            DatasetReaderInner::Tar(r) => r.payload_field(idx, field),
            DatasetReaderInner::Parquet(r) => r.payload_field(idx, field),
            DatasetReaderInner::Partitioned(r) => r.payload_field(idx, field),
            _ => bail!("dataset does not contain payloads"),
        }
    }

    pub fn payload_object(&self, idx: usize) -> Result<Option<Value>> {
        match &self.inner {
            DatasetReaderInner::Tar(r) => r.payload_object(idx),
            DatasetReaderInner::Parquet(r) => r.payload_object(idx),
            DatasetReaderInner::Partitioned(r) => r.payload_object(idx),
            _ => bail!("dataset does not contain payloads"),
        }
    }

    /// Number of queries available in the dataset's query set (0 if none).
    pub fn num_queries(&self) -> usize {
        match &self.inner {
            DatasetReaderInner::H5(r) => r.num_queries(),
            DatasetReaderInner::Tar(r) => r.num_queries(),
            DatasetReaderInner::Sparse(r) => r.num_queries(),
            // Component formats hold corpus rows only; a query set is a
            // separate file, declared as its own source.
            DatasetReaderInner::Npy(_)
            | DatasetReaderInner::Parquet(_)
            | DatasetReaderInner::Partitioned(_)
            | DatasetReaderInner::Multivector(_) => 0,
        }
    }

    /// A dense query vector from the dataset's query set.
    pub fn query_dense_vector(&self, idx: usize) -> Result<Vec<f32>> {
        match &self.inner {
            DatasetReaderInner::H5(r) => r.query_at(idx),
            DatasetReaderInner::Tar(r) => r.query_at(idx),
            DatasetReaderInner::Sparse(_) => bail!("sparse dataset has no dense queries"),
            DatasetReaderInner::Npy(_)
            | DatasetReaderInner::Parquet(_)
            | DatasetReaderInner::Partitioned(_)
            | DatasetReaderInner::Multivector(_) => bail!("dataset has no query set"),
        }
    }

    /// A sparse query vector from the dataset's query set.
    pub fn query_sparse_vector(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        match &self.inner {
            DatasetReaderInner::Sparse(r) => r.query_at(idx),
            _ => bail!("dataset does not contain sparse queries"),
        }
    }

    /// The whole dense query set with its ground truth, read in one pass.
    ///
    /// Text-backed formats get a sequential fast path; the rest fall back to
    /// indexed reads, which for a binary format cost the same either way.
    pub fn read_dense_query_set(&self) -> Result<Vec<QueryEntry<Vec<f32>>>> {
        if let DatasetReaderInner::Tar(reader) = &self.inner {
            return reader.read_query_set();
        }
        let mut rows = Vec::with_capacity(self.num_queries());
        for idx in 0..self.num_queries() {
            rows.push(QueryEntry {
                vector: self.query_dense_vector(idx)?,
                ground_truth: self.query_ground_truth(idx)?,
                // Only the tar layout has a `conditions` field.
                conditions: None,
            });
        }
        Ok(rows)
    }

    /// The whole sparse query set with its ground truth. Sparse queries live in
    /// binary CSR files, so there is nothing to gain from a bulk path.
    pub fn read_sparse_query_set(&self) -> Result<Vec<QueryEntry<SparseVector>>> {
        let mut rows = Vec::with_capacity(self.num_queries());
        for idx in 0..self.num_queries() {
            rows.push(QueryEntry {
                vector: self.query_sparse_vector(idx)?,
                ground_truth: self.query_ground_truth(idx)?,
                conditions: None,
            });
        }
        Ok(rows)
    }

    /// Ground-truth nearest-neighbor ids for a query (indices into the corpus).
    pub fn query_ground_truth(&self, idx: usize) -> Result<Vec<u64>> {
        match &self.inner {
            DatasetReaderInner::H5(r) => r.neighbors_at(idx),
            DatasetReaderInner::Tar(r) => r.query_ground_truth(idx),
            DatasetReaderInner::Sparse(r) => r.query_ground_truth(idx),
            DatasetReaderInner::Npy(_)
            | DatasetReaderInner::Parquet(_)
            | DatasetReaderInner::Partitioned(_)
            | DatasetReaderInner::Multivector(_) => bail!("dataset has no ground truth"),
        }
    }
}
