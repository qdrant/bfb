use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ndarray::Axis;

pub struct H5Reader {
    file: Arc<hdf5::File>,
    num_points: usize,
}

impl H5Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = Arc::new(
            hdf5::File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
        );
        let dataset = file
            .dataset("train")
            .context("h5 dataset is missing `train` dataset")?;
        let shape = dataset.shape();
        if shape.len() != 2 {
            bail!("expected `train` to be a 2-D dataset, got {shape:?}");
        }
        Ok(H5Reader {
            file,
            num_points: shape[0],
        })
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.num_points {
            bail!("index {idx} out of range (dataset has {} points)", self.num_points);
        }
        let dataset = self
            .file
            .dataset("train")
            .context("h5 dataset is missing `train` dataset")?;
        let row = dataset
            .read_slice::<f32, _, ndarray::Ix2>((idx..idx + 1, ..))
            .with_context(|| format!("failed to read vector {idx}"))?;
        Ok(row.index_axis(Axis(0), 0).iter().copied().collect())
    }
}
