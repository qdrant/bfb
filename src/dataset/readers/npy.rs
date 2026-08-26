//! `.npy` (NumPy array) reading.
//!
//! Two consumers share this module: the `tar` bundle format, whose payload is a
//! `vectors.npy` inside an extracted archive, and the standalone `npy` dataset
//! format, which points straight at one such file.
//!
//! Rows are served from an mmap, so access is lock-free and resident memory is
//! reclaimable page cache rather than committed RAM.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use memmap2::Mmap;

/// Element type of a `.npy` array (little-endian float).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }
}

/// Shape and element layout read from a `.npy` header.
#[derive(Debug, Clone, Copy)]
pub struct NpyLayout {
    pub dtype: Dtype,
    pub num_points: usize,
    pub dim: usize,
    /// Byte offset of the raw array data within the file.
    pub data_offset: usize,
}

/// A 2-D float `.npy` array served from an mmap.
pub struct NpyMatrix {
    mmap: Mmap,
    layout: NpyLayout,
}

impl NpyMatrix {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;

        let layout = parse_npy_header(&mmap)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let needed = layout.data_offset + layout.num_points * layout.dim * layout.dtype.size();
        if mmap.len() < needed {
            bail!(
                "{} is truncated: {} bytes, need {needed}",
                path.display(),
                mmap.len()
            );
        }

        Ok(NpyMatrix { mmap, layout })
    }

    pub fn rows(&self) -> usize {
        self.layout.num_points
    }

    pub fn row(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.layout.num_points {
            bail!(
                "index {idx} out of range (array has {} rows)",
                self.layout.num_points
            );
        }
        let row_bytes = self.layout.dim * self.layout.dtype.size();
        let start = self.layout.data_offset + idx * row_bytes;
        let bytes = &self.mmap[start..start + row_bytes];
        Ok(match self.layout.dtype {
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b).to_f32())
                .collect(),
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::F64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect(),
        })
    }
}

/// A standalone `.npy` file used as a dense-vector dataset source.
///
/// Vectors only: it carries no payloads, query set, or ground truth, so those
/// accessors are rejected by [`DatasetReader`](crate::dataset::DatasetReader).
pub struct NpyReader {
    matrix: NpyMatrix,
}

impl NpyReader {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(NpyReader {
            matrix: NpyMatrix::open(path)?,
        })
    }

    pub fn num_points(&self) -> usize {
        self.matrix.rows()
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        self.matrix.row(idx)
    }
}

/// Minimal parser for the `.npy` format (v1/v2 headers) sufficient for the 2-D
/// float arrays shipped by vector-db-benchmark and by embedding dumps such as
/// LAION's `img_emb_*.npy`.
///
/// Only the leading header is read, so `buf` may be a short prefix of the file
/// — which is what makes remote row counts a single ranged request rather than
/// a download.
pub fn parse_npy_header(buf: &[u8]) -> Result<NpyLayout> {
    let (header, header_end) = parse_npy_header_str(buf)?;

    let descr = extract_quoted(header, "descr").context(".npy header missing 'descr'")?;
    let dtype = match descr.as_str() {
        "<f2" | "|f2" => Dtype::F16,
        "<f4" | "|f4" => Dtype::F32,
        "<f8" | "|f8" => Dtype::F64,
        other => bail!("unsupported .npy dtype {other:?} (expected float16/32/64)"),
    };

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        bail!(".npy array is Fortran-ordered; expected C order");
    }

    let (num_points, dim) = extract_shape(header)?;
    Ok(NpyLayout {
        dtype,
        num_points,
        dim,
        data_offset: header_end,
    })
}

/// Parse the leading `.npy` magic/header framing, returning the header dict
/// string and the byte offset where the array data starts. Shared by
/// [`parse_npy_header`] (2-D float arrays) and the `offsets.npy` reader for
/// multivector datasets (1-D int arrays), which parse the returned dict
/// differently.
pub(crate) fn parse_npy_header_str(buf: &[u8]) -> Result<(&str, usize)> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        bail!("not a .npy file (bad magic)");
    }
    let major = buf[6];
    // Header length field: 2 bytes (v1) or 4 bytes (v2+), little-endian.
    let (header_len, header_start) = if major >= 2 {
        if buf.len() < 12 {
            bail!("truncated .npy header");
        }
        (
            u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize,
            12,
        )
    } else {
        (
            u16::from_le_bytes(buf[8..10].try_into().unwrap()) as usize,
            10,
        )
    };
    let header_end = header_start + header_len;
    if buf.len() < header_end {
        bail!(
            "truncated .npy header: need {header_end} bytes, got {}",
            buf.len()
        );
    }
    let header = std::str::from_utf8(&buf[header_start..header_end])
        .context(".npy header is not valid UTF-8")?;
    Ok((header, header_end))
}

/// Extract a single-quoted string value for `key` from a `.npy` header dict.
pub(crate) fn extract_quoted(header: &str, key: &str) -> Option<String> {
    let after_key = &header[header.find(&format!("'{key}'"))?..];
    let after_colon = &after_key[after_key.find(':')? + 1..];
    let bytes = after_colon.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || (bytes[i] != b'\'' && bytes[i] != b'"') {
        return None;
    }
    let quote = bytes[i];
    i += 1;
    let start = i;
    while i < bytes.len() && bytes[i] != quote {
        i += 1;
    }
    Some(after_colon[start..i].to_string())
}

/// Extract the 2-D `(rows, cols)` shape tuple from a `.npy` header dict.
fn extract_shape(header: &str) -> Result<(usize, usize)> {
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
    if dims.len() != 2 {
        bail!("expected a 2-D .npy array, got shape {dims:?}");
    }
    Ok((dims[0], dims[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::fixtures::{make_npy, make_ramp_npy};

    #[test]
    fn reads_f32_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.npy");
        std::fs::write(&path, make_ramp_npy(0, 2, 3)).unwrap();

        let reader = NpyReader::open(&path).unwrap();
        assert_eq!(reader.num_points(), 2);
        assert_eq!(reader.vector_at(1).unwrap(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn reads_f16_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.npy");
        let values = [0.5f32, -1.0, 2.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        std::fs::write(&path, make_npy("<f2", 2, 2, &bytes)).unwrap();

        let reader = NpyReader::open(&path).unwrap();
        assert_eq!(reader.num_points(), 2);
        assert_eq!(reader.vector_at(0).unwrap(), vec![0.5, -1.0]);
        assert_eq!(reader.vector_at(1).unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn rejects_out_of_range_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.npy");
        std::fs::write(&path, make_ramp_npy(0, 2, 3)).unwrap();

        let reader = NpyReader::open(&path).unwrap();
        assert!(reader.vector_at(2).is_err());
    }

    /// The header must be parseable from a short prefix — this is what lets a
    /// remote part be sized with one ranged request instead of a download.
    #[test]
    fn parses_header_from_a_prefix_of_the_file() {
        let full = make_ramp_npy(0, 1_000_448, 512);
        let layout = parse_npy_header(&full[..256]).unwrap();
        assert_eq!(layout.num_points, 1_000_448);
        assert_eq!(layout.dim, 512);
        assert_eq!(layout.dtype, Dtype::F32);
    }

    #[test]
    fn reports_a_prefix_too_short_to_hold_the_header() {
        let full = make_ramp_npy(0, 4, 4);
        let err = parse_npy_header(&full[..12]).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn rejects_non_npy_and_unsupported_dtypes() {
        assert!(parse_npy_header(b"not an npy file at all").is_err());

        let ints: Vec<u8> = (0..4u32).flat_map(|x| x.to_le_bytes()).collect();
        let err = parse_npy_header(&make_npy("<i4", 2, 2, &ints)).unwrap_err();
        assert!(err.to_string().contains("unsupported"), "{err}");
    }
}
