use std::path::Path;

use anyhow::{Context, Result, bail};
use hdf5_pure_rust::File;

pub struct H5Reader {
    /// Full `train` matrix in row-major order (`num_points` × `dim`).
    data: Vec<f32>,
    dim: usize,
    num_points: usize,
}

impl H5Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let ds = file
            .dataset("train")
            .context("h5 dataset is missing `train` dataset")?;
        let shape = ds.shape().context("failed to read `train` shape")?;
        if shape.len() != 2 {
            bail!("expected `train` to be a 2-D dataset, got {shape:?}");
        }
        let num_points = shape[0] as usize;
        let dim = shape[1] as usize;
        // Read the whole dataset once. ann-benchmarks `train` datasets are
        // chunked+gzipped, so a per-row read re-decompresses the enclosing chunk
        // on every access; a single sequential read decompresses each chunk once
        // and lets `vector_at` serve rows lock-free from memory.
        let data = ds
            .read::<f32>()
            .context("failed to read `train` dataset")?;
        if data.len() != num_points * dim {
            bail!(
                "`train` has {} elements, expected {} ({num_points}×{dim})",
                data.len(),
                num_points * dim
            );
        }
        Ok(H5Reader {
            data,
            dim,
            num_points,
        })
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.num_points {
            bail!(
                "index {idx} out of range (dataset has {} points)",
                self.num_points
            );
        }
        let start = idx * self.dim;
        Ok(self.data[start..start + self.dim].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_pure_rust::WritableFile;

    #[test]
    fn read_train_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");
        {
            let mut wf = WritableFile::create(&path).unwrap();
            let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
            wf.new_dataset_builder("train")
                .shape(&[3, 4])
                .write(&data)
                .unwrap();
            wf.close().unwrap();
        }

        let reader = H5Reader::open(&path).unwrap();
        assert_eq!(reader.num_points(), 3);
        assert_eq!(reader.vector_at(1).unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
    }
}
