use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use memmap2::Mmap;
use serde_json::Value;

use super::jsonl::JsonlStore;

/// Element type of a `vectors.npy` array (little-endian float).
#[derive(Debug, Clone, Copy)]
enum Dtype {
    F16,
    F32,
    F64,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }
}

pub struct TarReader {
    mmap: Mmap,
    dtype: Dtype,
    num_points: usize,
    dim: usize,
    /// Byte offset of the raw array data within the mmap.
    data_offset: usize,
    payloads: Option<JsonlStore>,
}

impl TarReader {
    pub fn open(path: &Path) -> Result<Self> {
        let vectors_path = path.join("vectors.npy");
        let file = File::open(&vectors_path)
            .with_context(|| format!("failed to open {}", vectors_path.display()))?;
        let mmap = unsafe { Mmap::map(&file).context("failed to mmap vectors.npy")? };

        let layout = parse_npy_header(&mmap)
            .with_context(|| format!("failed to parse {}", vectors_path.display()))?;
        let needed = layout.data_offset + layout.num_points * layout.dim * layout.dtype.size();
        if mmap.len() < needed {
            bail!(
                "{} is truncated: {} bytes, need {needed}",
                vectors_path.display(),
                mmap.len()
            );
        }

        let payloads_path = path.join("payloads.jsonl");
        let payloads = if payloads_path.exists() {
            Some(JsonlStore::open(&payloads_path)?)
        } else {
            None
        };

        Ok(TarReader {
            mmap,
            dtype: layout.dtype,
            num_points: layout.num_points,
            dim: layout.dim,
            data_offset: layout.data_offset,
            payloads,
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
        let row_bytes = self.dim * self.dtype.size();
        let start = self.data_offset + idx * row_bytes;
        let bytes = &self.mmap[start..start + row_bytes];
        Ok(match self.dtype {
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F64 => bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
        })
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        let Some(store) = &self.payloads else {
            return Ok(None);
        };
        let line = store
            .value_at(idx)?
            .with_context(|| format!("payload index {idx} out of range"))?;
        Ok(line.get(field).cloned())
    }

    /// The whole payload object for a point (the full `payloads.jsonl` line).
    pub fn payload_object(&self, idx: usize) -> Result<Option<Value>> {
        let Some(store) = &self.payloads else {
            return Ok(None);
        };
        store.value_at(idx)
    }
}

struct NpyLayout {
    dtype: Dtype,
    num_points: usize,
    dim: usize,
    data_offset: usize,
}

/// Minimal parser for the `.npy` format (v1/v2 headers) sufficient for the 2-D
/// float `vectors.npy` files shipped by vector-db-benchmark.
fn parse_npy_header(buf: &[u8]) -> Result<NpyLayout> {
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
        bail!("truncated .npy header");
    }
    let header = std::str::from_utf8(&buf[header_start..header_end])
        .context(".npy header is not valid UTF-8")?;

    let descr = extract_quoted(header, "descr").context(".npy header missing 'descr'")?;
    let dtype = match descr.as_str() {
        "<f2" | "|f2" => Dtype::F16,
        "<f4" | "|f4" => Dtype::F32,
        "<f8" | "|f8" => Dtype::F64,
        other => bail!("unsupported vectors.npy dtype {other:?} (expected float16/32/64)"),
    };

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        bail!("vectors.npy is Fortran-ordered; expected C order");
    }

    let (num_points, dim) = extract_shape(header)?;
    Ok(NpyLayout {
        dtype,
        num_points,
        dim,
        data_offset: header_end,
    })
}

/// Extract a single-quoted string value for `key` from a `.npy` header dict.
fn extract_quoted(header: &str, key: &str) -> Option<String> {
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
        bail!("expected 2-D vectors.npy, got shape {dims:?}");
    }
    Ok((dims[0], dims[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal little-endian `.npy` v1.0 buffer.
    fn make_npy(descr: &str, rows: usize, cols: usize, data: &[u8]) -> Vec<u8> {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        let mut header = dict.into_bytes();
        // Pad with spaces so the total (magic + version + len field + header)
        // is a multiple of 64, and terminate with a newline.
        let unpadded = 10 + header.len() + 1;
        header.extend(std::iter::repeat(b' ').take((64 - unpadded % 64) % 64));
        header.push(b'\n');

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.extend_from_slice(&[1, 0]); // version 1.0
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out
    }

    fn write_dataset(dir: &Path, npy: &[u8]) {
        std::fs::write(dir.join("vectors.npy"), npy).unwrap();
    }

    #[test]
    fn reads_f32_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_dataset(dir.path(), &make_npy("<f4", 2, 3, &bytes));

        let reader = TarReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 2);
        assert_eq!(reader.vector_at(1).unwrap(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn reads_f16_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let values = [0.5f32, -1.0, 2.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        write_dataset(dir.path(), &make_npy("<f2", 2, 2, &bytes));

        let reader = TarReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 2);
        assert_eq!(reader.vector_at(0).unwrap(), vec![0.5, -1.0]);
        assert_eq!(reader.vector_at(1).unwrap(), vec![2.0, 4.0]);
    }
}
