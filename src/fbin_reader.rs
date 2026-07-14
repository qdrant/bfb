use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;

#[derive(Debug)]
pub struct FBinReader {
    pub num_vectors: i32,
    pub dim: i32,
    pub iter_offset: usize,
    pub header_size: usize,
    mmap: Mmap,
}

impl FBinReader {
    pub fn new(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .append(false)
            .create(false)
            .open(path)
            .with_context(|| format!("failed to open vector file {}", path.display()))?;

        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap vector file {}", path.display()))?;

        let int_size = size_of::<i32>();
        let dim_offset = int_size;
        let header_size = dim_offset + int_size;
        if mmap.len() < header_size {
            bail!(
                "vector file {} is too short to hold an fbin header",
                path.display()
            );
        }
        let num_vectors_raw = &mmap[0..dim_offset];
        let num_dim_raw = &mmap[dim_offset..header_size];

        let num_vectors = i32::from_le_bytes(num_vectors_raw.try_into().unwrap());
        let dim = i32::from_le_bytes(num_dim_raw.try_into().unwrap());

        if num_vectors <= 0 || dim <= 0 {
            bail!(
                "vector file {} has an invalid fbin header: num_vectors={num_vectors}, dim={dim}",
                path.display()
            );
        }

        // The declared payload must fit in the file, otherwise `read_vector` would read
        // past the end of the mapping on a truncated or malformed file.
        let expected_size = (num_vectors as usize)
            .checked_mul(dim as usize)
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .and_then(|payload| payload.checked_add(header_size))
            .with_context(|| {
                format!(
                    "vector file {} declares an unrepresentable size: {num_vectors} x {dim}",
                    path.display()
                )
            })?;
        if mmap.len() < expected_size {
            bail!(
                "vector file {} is truncated: header declares {num_vectors} vectors of {dim} \
                 dimensions ({expected_size} bytes), but the file is {} bytes",
                path.display(),
                mmap.len(),
            );
        }

        Ok(FBinReader {
            num_vectors,
            dim,
            iter_offset: 0,
            header_size,
            mmap,
        })
    }

    /// Vector at `idx`. Panics if `idx >= num_vectors`.
    pub fn read_vector(&self, idx: usize) -> &[f32] {
        let dim = self.dim as usize;
        let vector_size = dim * size_of::<f32>();
        let vector_offset = self.header_size + idx * vector_size;
        let vector_raw = &self.mmap[vector_offset..vector_offset + vector_size];
        // SAFETY: the slice above is bounds-checked and holds exactly `dim` f32 values.
        // It is 4-byte aligned because the mmap base is page aligned, the header is 8
        // bytes and every vector is a multiple of 4 bytes.
        unsafe { std::slice::from_raw_parts(vector_raw.as_ptr().cast::<f32>(), dim) }
    }
}

impl Iterator for FBinReader {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_offset >= self.num_vectors as usize {
            return None;
        }

        let vector = self.read_vector(self.iter_offset).to_vec();
        self.iter_offset += 1;
        Some(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write an fbin file with the given header, followed by `payload` vector values.
    fn write_fbin(num_vectors: i32, dim: i32, payload: &[f32]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&num_vectors.to_le_bytes()).unwrap();
        file.write_all(&dim.to_le_bytes()).unwrap();
        for value in payload {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn reads_vectors_of_a_valid_file() {
        let file = write_fbin(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reader = FBinReader::new(file.path()).unwrap();
        assert_eq!(reader.num_vectors, 2);
        assert_eq!(reader.dim, 3);
        assert_eq!(reader.read_vector(0), [1.0, 2.0, 3.0]);
        assert_eq!(reader.read_vector(1), [4.0, 5.0, 6.0]);
        assert_eq!(reader.count(), 2);
    }

    #[test]
    fn rejects_truncated_payload() {
        // Header promises 4 vectors, only 1 is present.
        let file = write_fbin(4, 3, &[1.0, 2.0, 3.0]);
        let err = FBinReader::new(file.path()).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn rejects_negative_header_values() {
        let file = write_fbin(-1, 3, &[]);
        let err = FBinReader::new(file.path()).unwrap_err();
        assert!(err.to_string().contains("invalid fbin header"), "{err}");

        let file = write_fbin(1, -3, &[]);
        let err = FBinReader::new(file.path()).unwrap_err();
        assert!(err.to_string().contains("invalid fbin header"), "{err}");
    }

    #[test]
    fn rejects_header_that_overflows_the_file_size() {
        // 2^31-1 vectors of 2^31-1 dimensions cannot fit in any file.
        let file = write_fbin(i32::MAX, i32::MAX, &[]);
        let err = FBinReader::new(file.path()).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn rejects_file_shorter_than_the_header() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0u8; 4]).unwrap();
        file.flush().unwrap();
        let err = FBinReader::new(file.path()).unwrap_err();
        assert!(err.to_string().contains("too short"), "{err}");
    }
}
