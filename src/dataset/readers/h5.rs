use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use hdf5_pure_rust::File;

pub struct H5Reader {
    file: Mutex<File>,
    dim: usize,
    num_points: usize,
}

impl H5Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let ds = file
            .dataset("train")
            .context("h5 dataset is missing `train` dataset")?;
        let shape = ds.shape().context("failed to read `train` shape")?;
        if shape.len() != 2 {
            bail!("expected `train` to be a 2-D dataset, got {shape:?}");
        }
        let num_points = shape[0] as usize;
        let dim = shape[1] as usize;
        Ok(H5Reader {
            file: Mutex::new(file),
            dim,
            num_points,
        })
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.num_points {
            bail!("index {idx} out of range (dataset has {} points)", self.num_points);
        }
        let file = self.file.lock().expect("h5 file mutex poisoned");
        let ds = file
            .dataset("train")
            .context("h5 dataset is missing `train` dataset")?;
        let mut row = vec![0f32; self.dim];
        ds.read_slice_into::<f32, _>((idx..idx + 1, ..), &mut row)
            .with_context(|| format!("failed to read vector {idx}"))?;
        Ok(row)
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
