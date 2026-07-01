use std::fs::File as FsFile;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hdf5_pure_rust::File;
use memmap2::Mmap;

/// Rows are decompressed once into a flat sidecar file and served from an mmap,
/// so `vector_at` is lock-free and resident memory is reclaimable page cache
/// rather than committed RAM.
pub struct H5Reader {
    mmap: Mmap,
    dim: usize,
    num_points: usize,
}

/// Target size of the row-band decompressed at a time while building the
/// sidecar (~16 MiB), bounding peak memory during conversion.
const BAND_BYTES: usize = 16 * 1024 * 1024;

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
        let expected_bytes = num_points * dim * std::mem::size_of::<f32>();

        // ann-benchmarks `train` datasets are chunked+gzipped, so a per-row read
        // re-decompresses the enclosing chunk on every access. Decompress once
        // into a flat little-endian `f32` sidecar and mmap it; reuse the sidecar
        // across runs when it already matches the expected size.
        let sidecar = sidecar_path(path);
        let needs_build = match std::fs::metadata(&sidecar) {
            Ok(meta) => meta.len() != expected_bytes as u64,
            Err(_) => true,
        };
        if needs_build {
            build_sidecar(&ds, &sidecar, num_points, dim)?;
        }

        let file = FsFile::open(&sidecar)
            .with_context(|| format!("failed to open {}", sidecar.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", sidecar.display()))?;
        if mmap.len() != expected_bytes {
            bail!(
                "{} has {} bytes, expected {expected_bytes} ({num_points}×{dim} f32)",
                sidecar.display(),
                mmap.len()
            );
        }

        Ok(H5Reader {
            mmap,
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
        let start = idx * self.dim * std::mem::size_of::<f32>();
        let end = start + self.dim * std::mem::size_of::<f32>();
        Ok(self.mmap[start..end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

/// Flat sidecar path for an h5 file, e.g. `foo.hdf5` -> `foo.train.f32`.
fn sidecar_path(path: &Path) -> PathBuf {
    path.with_extension("train.f32")
}

/// Decompress the `train` dataset into a flat little-endian `f32` file, reading
/// one row band at a time so peak memory stays bounded. Written to a temp file
/// and atomically renamed so a crash never leaves a partial-but-right-size file.
fn build_sidecar(
    ds: &hdf5_pure_rust::Dataset,
    sidecar: &Path,
    num_points: usize,
    dim: usize,
) -> Result<()> {
    let dir = sidecar.parent().unwrap_or_else(|| Path::new("."));
    let band_rows = (BAND_BYTES / (dim * std::mem::size_of::<f32>())).max(1);

    let tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create temp file in {}", dir.display()))?;
    let mut writer = BufWriter::new(tmp.reopen().context("failed to reopen temp file")?);

    let mut floats = vec![0f32; band_rows * dim];
    let mut bytes = Vec::with_capacity(band_rows * dim * std::mem::size_of::<f32>());
    let mut start = 0;
    while start < num_points {
        let end = (start + band_rows).min(num_points);
        let slice = &mut floats[..(end - start) * dim];
        ds.read_slice_into::<f32, _>((start..end, ..), slice)
            .with_context(|| format!("failed to read `train` rows {start}..{end}"))?;
        bytes.clear();
        for value in slice.iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes).context("failed to write sidecar")?;
        start = end;
    }
    writer.flush().context("failed to flush sidecar")?;
    drop(writer);

    tmp.persist(sidecar)
        .with_context(|| format!("failed to persist {}", sidecar.display()))?;
    Ok(())
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

        // Second open reuses the freshly-built sidecar.
        let reader = H5Reader::open(&path).unwrap();
        assert_eq!(reader.vector_at(2).unwrap(), vec![8.0, 9.0, 10.0, 11.0]);
    }
}
