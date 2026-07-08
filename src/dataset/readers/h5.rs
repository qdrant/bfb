use std::fs::File as FsFile;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hdf5_pure_rust::File;
use memmap2::Mmap;

/// A 2-D array decompressed once into a flat little-endian sidecar and served
/// from an mmap, so element access is lock-free and resident memory is
/// reclaimable page cache rather than committed RAM.
struct MmapMatrix {
    mmap: Mmap,
    cols: usize,
    rows: usize,
    elem_size: usize,
}

/// Rows are decompressed once into a flat sidecar file and served from an mmap,
/// so `vector_at` is lock-free and resident memory is reclaimable page cache
/// rather than committed RAM.
///
/// ann-benchmarks HDF5 files contain up to four datasets:
/// * `train` — corpus vectors (always required),
/// * `test` — query vectors (optional; needed to use the dataset as a query
///   source),
/// * `neighbors` — ground-truth nearest-neighbor indices for each query
///   (optional; needed to measure search accuracy).
pub struct H5Reader {
    train: MmapMatrix,
    test: Option<MmapMatrix>,
    neighbors: Option<MmapMatrix>,
}

/// Target size of the row-band decompressed at a time while building the
/// sidecar (~16 MiB), bounding peak memory during conversion.
const BAND_BYTES: usize = 16 * 1024 * 1024;

impl H5Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;

        let train = open_matrix_f32(&file, path, "train")?
            .context("h5 dataset is missing `train` dataset")?;
        let test = open_matrix_f32(&file, path, "test")?;
        let neighbors = open_matrix_i64(&file, path, "neighbors")?;

        Ok(H5Reader {
            train,
            test,
            neighbors,
        })
    }

    pub fn num_points(&self) -> usize {
        self.train.rows
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        self.train.row_f32(idx, "train")
    }

    /// Number of query vectors in the `test` dataset (0 if absent).
    pub fn num_queries(&self) -> usize {
        self.test.as_ref().map_or(0, |m| m.rows)
    }

    /// A query vector from the `test` dataset.
    pub fn query_at(&self, idx: usize) -> Result<Vec<f32>> {
        let test = self
            .test
            .as_ref()
            .context("h5 dataset is missing `test` dataset (no query vectors)")?;
        test.row_f32(idx, "test")
    }

    /// Ground-truth nearest-neighbor ids for a query, from `neighbors`.
    pub fn neighbors_at(&self, idx: usize) -> Result<Vec<u64>> {
        let neighbors = self
            .neighbors
            .as_ref()
            .context("h5 dataset is missing `neighbors` dataset (no ground truth)")?;
        Ok(neighbors
            .row_i64(idx, "neighbors")?
            .into_iter()
            .map(|v| v as u64)
            .collect())
    }
}

impl MmapMatrix {
    fn row_bytes(&self, idx: usize, name: &str) -> Result<&[u8]> {
        if idx >= self.rows {
            bail!("index {idx} out of range (`{name}` has {} rows)", self.rows);
        }
        let start = idx * self.cols * self.elem_size;
        let end = start + self.cols * self.elem_size;
        Ok(&self.mmap[start..end])
    }

    fn row_f32(&self, idx: usize, name: &str) -> Result<Vec<f32>> {
        Ok(self
            .row_bytes(idx, name)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn row_i64(&self, idx: usize, name: &str) -> Result<Vec<i64>> {
        Ok(self
            .row_bytes(idx, name)?
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

/// Open a 2-D dataset as an `f32` sidecar-backed matrix. Returns `Ok(None)` if
/// the dataset does not exist in the file.
fn open_matrix_f32(file: &File, path: &Path, name: &str) -> Result<Option<MmapMatrix>> {
    let Ok(ds) = file.dataset(name) else {
        return Ok(None);
    };
    let (rows, cols) = shape_2d(&ds, name)?;
    let sidecar = sidecar_path(path, name, "f32");
    let expected_bytes = rows * cols * std::mem::size_of::<f32>();
    if needs_build(&sidecar, expected_bytes) {
        build_sidecar_f32(&ds, &sidecar, rows, cols, name)?;
    }
    let mmap = open_sidecar(&sidecar, expected_bytes, rows, cols)?;
    Ok(Some(MmapMatrix {
        mmap,
        cols,
        rows,
        elem_size: std::mem::size_of::<f32>(),
    }))
}

/// Open a 2-D integer dataset as an `i64` sidecar-backed matrix (the stored
/// element type is converted to `i64` on read). Returns `Ok(None)` if absent.
fn open_matrix_i64(file: &File, path: &Path, name: &str) -> Result<Option<MmapMatrix>> {
    let Ok(ds) = file.dataset(name) else {
        return Ok(None);
    };
    let (rows, cols) = shape_2d(&ds, name)?;
    let sidecar = sidecar_path(path, name, "i64");
    let expected_bytes = rows * cols * std::mem::size_of::<i64>();
    if needs_build(&sidecar, expected_bytes) {
        build_sidecar_i64(&ds, &sidecar, rows, cols, name)?;
    }
    let mmap = open_sidecar(&sidecar, expected_bytes, rows, cols)?;
    Ok(Some(MmapMatrix {
        mmap,
        cols,
        rows,
        elem_size: std::mem::size_of::<i64>(),
    }))
}

fn shape_2d(ds: &hdf5_pure_rust::Dataset, name: &str) -> Result<(usize, usize)> {
    let shape = ds
        .shape()
        .with_context(|| format!("failed to read `{name}` shape"))?;
    if shape.len() != 2 {
        bail!("expected `{name}` to be a 2-D dataset, got {shape:?}");
    }
    Ok((shape[0] as usize, shape[1] as usize))
}

fn needs_build(sidecar: &Path, expected_bytes: usize) -> bool {
    match std::fs::metadata(sidecar) {
        Ok(meta) => meta.len() != expected_bytes as u64,
        Err(_) => true,
    }
}

fn open_sidecar(sidecar: &Path, expected_bytes: usize, rows: usize, cols: usize) -> Result<Mmap> {
    let file =
        FsFile::open(sidecar).with_context(|| format!("failed to open {}", sidecar.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap {}", sidecar.display()))?;
    if mmap.len() != expected_bytes {
        bail!(
            "{} has {} bytes, expected {expected_bytes} ({rows}×{cols})",
            sidecar.display(),
            mmap.len()
        );
    }
    Ok(mmap)
}

/// Flat sidecar path for an h5 dataset, e.g. `foo.hdf5` -> `foo.train.f32`.
fn sidecar_path(path: &Path, name: &str, ext: &str) -> PathBuf {
    path.with_extension(format!("{name}.{ext}"))
}

/// Decompress a 2-D dataset into a flat little-endian `f32` file, reading one
/// row band at a time so peak memory stays bounded. Written to a temp file and
/// atomically renamed so a crash never leaves a partial-but-right-size file.
fn build_sidecar_f32(
    ds: &hdf5_pure_rust::Dataset,
    sidecar: &Path,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<()> {
    build_sidecar(
        sidecar,
        rows,
        cols,
        std::mem::size_of::<f32>(),
        |band, start, end| {
            let mut floats = vec![0f32; (end - start) * cols];
            ds.read_slice_into::<f32, _>((start..end, ..), &mut floats)
                .with_context(|| format!("failed to read `{name}` rows {start}..{end}"))?;
            band.clear();
            for value in &floats {
                band.extend_from_slice(&value.to_le_bytes());
            }
            Ok(())
        },
    )
}

/// Like [`build_sidecar_f32`] but converts the stored (possibly int32) element
/// type to little-endian `i64`.
fn build_sidecar_i64(
    ds: &hdf5_pure_rust::Dataset,
    sidecar: &Path,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<()> {
    build_sidecar(
        sidecar,
        rows,
        cols,
        std::mem::size_of::<i64>(),
        |band, start, end| {
            let mut ints = vec![0i64; (end - start) * cols];
            ds.read_slice_into::<i64, _>((start..end, ..), &mut ints)
                .with_context(|| format!("failed to read `{name}` rows {start}..{end}"))?;
            band.clear();
            for value in &ints {
                band.extend_from_slice(&value.to_le_bytes());
            }
            Ok(())
        },
    )
}

/// Shared band-at-a-time sidecar writer. `fill_band` writes the little-endian
/// bytes for rows `start..end` into the reused `band` buffer.
fn build_sidecar(
    sidecar: &Path,
    rows: usize,
    cols: usize,
    elem_size: usize,
    mut fill_band: impl FnMut(&mut Vec<u8>, usize, usize) -> Result<()>,
) -> Result<()> {
    let dir = sidecar.parent().unwrap_or_else(|| Path::new("."));
    let band_rows = (BAND_BYTES / (cols.max(1) * elem_size)).max(1);

    let tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create temp file in {}", dir.display()))?;
    let mut writer = BufWriter::new(tmp.reopen().context("failed to reopen temp file")?);

    let mut band = Vec::with_capacity(band_rows * cols * elem_size);
    let mut start = 0;
    while start < rows {
        let end = (start + band_rows).min(rows);
        fill_band(&mut band, start, end)?;
        writer.write_all(&band).context("failed to write sidecar")?;
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
        assert_eq!(reader.num_queries(), 0);

        // Second open reuses the freshly-built sidecar.
        let reader = H5Reader::open(&path).unwrap();
        assert_eq!(reader.vector_at(2).unwrap(), vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn read_queries_and_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ann.h5");
        {
            let mut wf = WritableFile::create(&path).unwrap();
            let train: Vec<f32> = (0..12).map(|x| x as f32).collect();
            wf.new_dataset_builder("train")
                .shape(&[3, 4])
                .write(&train)
                .unwrap();
            let test: Vec<f32> = (0..8).map(|x| x as f32).collect();
            wf.new_dataset_builder("test")
                .shape(&[2, 4])
                .write(&test)
                .unwrap();
            let neighbors: Vec<i32> = vec![0, 2, 1, 2, 0, 1];
            wf.new_dataset_builder("neighbors")
                .shape(&[2, 3])
                .write(&neighbors)
                .unwrap();
            wf.close().unwrap();
        }

        let reader = H5Reader::open(&path).unwrap();
        assert_eq!(reader.num_points(), 3);
        assert_eq!(reader.num_queries(), 2);
        assert_eq!(reader.query_at(1).unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(reader.neighbors_at(0).unwrap(), vec![0, 2, 1]);
        assert_eq!(reader.neighbors_at(1).unwrap(), vec![2, 0, 1]);
    }
}
