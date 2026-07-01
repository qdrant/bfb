use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use super::config::{DatasetConfig, DatasetKind};
use super::download::ensure_downloaded;
use super::readers::{H5Reader, SparseReader, TarReader};
use super::registry::load_registry;

enum DatasetReaderInner {
    H5(H5Reader),
    Tar(TarReader),
    Sparse(SparseReader),
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
        };
        Ok(DatasetReader {
            inner,
            num_points,
        })
    }

    pub fn dense_vector(&self, idx: usize) -> Result<Vec<f32>> {
        match &self.inner {
            DatasetReaderInner::H5(r) => r.vector_at(idx),
            DatasetReaderInner::Tar(r) => r.vector_at(idx),
            DatasetReaderInner::Sparse(_) => bail!("dataset does not contain dense vectors"),
        }
    }

    pub fn sparse_vector(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        match &self.inner {
            DatasetReaderInner::Sparse(r) => r.vector_at(idx),
            _ => bail!("dataset does not contain sparse vectors"),
        }
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        match &self.inner {
            DatasetReaderInner::Tar(r) => r.payload_field(idx, field),
            _ => bail!("dataset does not contain payloads"),
        }
    }
}
