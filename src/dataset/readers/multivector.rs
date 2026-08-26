//! Reads a ColBERT-style multivector dataset: a directory holding
//! `vectors.npy` (a flat `[total_subvectors, dim]` float array, in the same
//! format as the plain `npy` dataset) and `offsets.npy` (a 1-D int array of
//! length `num_points + 1` giving row boundaries into `vectors.npy`).
//!
//! Point `i`'s sub-vectors are `vectors[offsets[i]:offsets[i+1]]`, mirroring
//! how [`SparseReader`](super::SparseReader)'s CSR `index_pointer` addresses
//! ragged rows — except each unit here is a whole `dim`-wide row rather than
//! a single scalar.

use std::fs::File;
use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::npy::{NpyMatrix, extract_quoted, parse_npy_header_str};

pub struct MultivectorReader {
    vectors: NpyMatrix,
    /// Row boundaries into `vectors`, length `num_points + 1`.
    offsets: Vec<i64>,
}

impl MultivectorReader {
    pub fn open(path: &Path) -> Result<Self> {
        let vectors = NpyMatrix::open(&path.join("vectors.npy"))?;
        let offsets = read_offsets_npy(&path.join("offsets.npy"))?;

        if offsets.len() < 2 {
            bail!(
                "offsets.npy must have at least 2 entries (num_points + 1), got {}",
                offsets.len()
            );
        }
        let last = *offsets.last().unwrap();
        if last < 0 || last as usize != vectors.rows() {
            bail!(
                "offsets.npy's last entry ({last}) does not match vectors.npy's row count ({})",
                vectors.rows()
            );
        }

        Ok(MultivectorReader { vectors, offsets })
    }

    pub fn num_points(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<Vec<f32>>> {
        if idx + 1 >= self.offsets.len() {
            bail!(
                "index {idx} out of range (dataset has {} points)",
                self.num_points()
            );
        }
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        if start < 0 || end < start {
            bail!("offsets.npy is not non-decreasing at index {idx}");
        }
        (start as usize..end as usize)
            .map(|row| self.vectors.row(row))
            .collect()
    }
}

/// Minimal parser for a 1-D numeric `.npy` array (`int32`/`int64`, signed or
/// unsigned, or `float32`/`float64` downcast to `i64`), used for the
/// `offsets.npy` row-boundary array. Read in full rather than mmapped: it is
/// tiny (`num_points + 1` scalars) next to `vectors.npy`.
fn read_offsets_npy(path: &Path) -> Result<Vec<i64>> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let (header, header_end) = parse_npy_header_str(&buf)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let descr = extract_quoted(header, "descr").context(".npy header missing 'descr'")?;
    // Offsets are conceptually integers, but some exporters (e.g. numpy's
    // default float dtype) write them as floats; downcast those to i64 rather
    // than rejecting the file, since the values are still whole numbers.
    let elem = match descr.as_str() {
        "<i4" | "|i4" => OffsetElem::I32,
        "<u4" | "|u4" => OffsetElem::U32,
        "<i8" | "|i8" => OffsetElem::I64,
        "<u8" | "|u8" => OffsetElem::U64,
        "<f4" | "|f4" => OffsetElem::F32,
        "<f8" | "|f8" => OffsetElem::F64,
        other => bail!(
            "unsupported offsets dtype {other:?} (expected int32/int64/uint32/uint64/float32/float64)"
        ),
    };
    let elem_size = elem.size();

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        bail!("offsets.npy is Fortran-ordered; expected C order");
    }

    let n = extract_1d_shape(header)?;
    let data = &buf[header_end..];
    if data.len() < n * elem_size {
        bail!(
            "offsets.npy is truncated: need {} bytes of data, got {}",
            n * elem_size,
            data.len()
        );
    }

    Ok((0..n)
        .map(|i| elem.read(&data[i * elem_size..(i + 1) * elem_size]))
        .collect())
}

#[derive(Debug, Clone, Copy)]
enum OffsetElem {
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl OffsetElem {
    fn size(self) -> usize {
        match self {
            OffsetElem::I32 | OffsetElem::U32 | OffsetElem::F32 => 4,
            OffsetElem::I64 | OffsetElem::U64 | OffsetElem::F64 => 8,
        }
    }

    fn read(self, b: &[u8]) -> i64 {
        match self {
            OffsetElem::I32 => i32::from_le_bytes(b.try_into().unwrap()) as i64,
            OffsetElem::U32 => u32::from_le_bytes(b.try_into().unwrap()) as i64,
            OffsetElem::I64 => i64::from_le_bytes(b.try_into().unwrap()),
            OffsetElem::U64 => u64::from_le_bytes(b.try_into().unwrap()) as i64,
            OffsetElem::F32 => f32::from_le_bytes(b.try_into().unwrap()) as i64,
            OffsetElem::F64 => f64::from_le_bytes(b.try_into().unwrap()) as i64,
        }
    }
}

/// Extract the 1-D `(len,)` shape from a `.npy` header dict.
fn extract_1d_shape(header: &str) -> Result<usize> {
    let after_key = &header[header
        .find("'shape'")
        .context(".npy header missing 'shape'")?..];
    let open = after_key.find('(').context("malformed 'shape'")?;
    let close = after_key[open..].find(')').context("malformed 'shape'")? + open;
    let dims: Vec<usize> = after_key[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .context("malformed 'shape' dims")?;
    if dims.len() != 1 {
        bail!("expected a 1-D .npy array for offsets, got shape {dims:?}");
    }
    Ok(dims[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::fixtures::make_ramp_npy;

    fn make_offsets_npy(offsets: &[i64]) -> Vec<u8> {
        make_offsets_npy_dtype(offsets, "<i8", |o| o.to_le_bytes().to_vec())
    }

    fn make_f32_offsets_npy(offsets: &[i64]) -> Vec<u8> {
        make_offsets_npy_dtype(offsets, "<f4", |&o| (o as f32).to_le_bytes().to_vec())
    }

    fn make_offsets_npy_dtype(
        offsets: &[i64],
        descr: &str,
        encode: impl Fn(&i64) -> Vec<u8>,
    ) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '{descr}', 'fortran_order': False, 'shape': ({},), }}",
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
        for o in offsets {
            buf.extend_from_slice(&encode(o));
        }
        buf
    }

    #[test]
    fn reads_ragged_multivectors() {
        let dir = tempfile::tempdir().unwrap();
        // 5 sub-vectors total, dim 3: rows 0..5.
        std::fs::write(dir.path().join("vectors.npy"), make_ramp_npy(0, 5, 3)).unwrap();
        // Point 0 -> rows [0,2), point 1 -> [2,2) (empty), point 2 -> [2,5).
        std::fs::write(
            dir.path().join("offsets.npy"),
            make_offsets_npy(&[0, 2, 2, 5]),
        )
        .unwrap();

        let reader = MultivectorReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 3);
        assert_eq!(
            reader.vector_at(0).unwrap(),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
        );
        assert!(reader.vector_at(1).unwrap().is_empty());
        assert_eq!(reader.vector_at(2).unwrap().len(), 3);
    }

    /// Some exporters (e.g. numpy's default float dtype) write offsets as
    /// floats; they must be downcast to i64, not rejected.
    #[test]
    fn reads_float_offsets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vectors.npy"), make_ramp_npy(0, 5, 3)).unwrap();
        std::fs::write(
            dir.path().join("offsets.npy"),
            make_f32_offsets_npy(&[0, 2, 2, 5]),
        )
        .unwrap();

        let reader = MultivectorReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 3);
        assert_eq!(
            reader.vector_at(0).unwrap(),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
        );
    }

    #[test]
    fn rejects_offsets_mismatched_with_vector_count() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vectors.npy"), make_ramp_npy(0, 5, 3)).unwrap();
        std::fs::write(dir.path().join("offsets.npy"), make_offsets_npy(&[0, 2, 4])).unwrap();

        let err = match MultivectorReader::open(dir.path()) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("does not match"), "{err}");
    }

    #[test]
    fn rejects_out_of_range_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vectors.npy"), make_ramp_npy(0, 5, 3)).unwrap();
        std::fs::write(
            dir.path().join("offsets.npy"),
            make_offsets_npy(&[0, 2, 2, 5]),
        )
        .unwrap();

        let reader = MultivectorReader::open(dir.path()).unwrap();
        assert!(reader.vector_at(3).is_err());
    }
}
