use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::binary::{read_f32_array, read_i32_array, read_i64_array};

pub struct SparseReader {
    matrix: CsrMatrix,
}

impl SparseReader {
    pub fn open(path: &Path) -> Result<Self> {
        let data_path = path.join("data.csr");
        Ok(SparseReader {
            matrix: CsrMatrix::open(&data_path)?,
        })
    }

    pub fn num_points(&self) -> usize {
        self.matrix.num_rows
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        self.matrix.row_sparse(idx)
    }
}

struct CsrMatrix {
    num_rows: usize,
    index_pointer: Vec<i64>,
    columns: Vec<i32>,
    values: Vec<f32>,
}

impl CsrMatrix {
    fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut header = [0u8; 24];
        file.read_exact(&mut header)
            .context("failed to read csr header")?;
        let sizes: [i64; 3] = [
            i64::from_le_bytes(header[0..8].try_into().unwrap()),
            i64::from_le_bytes(header[8..16].try_into().unwrap()),
            i64::from_le_bytes(header[16..24].try_into().unwrap()),
        ];
        let n_row = sizes[0] as usize;
        let n_non_zero = sizes[2] as usize;

        let mut index_pointer = vec![0i64; n_row + 1];
        read_i64_array(&mut file, &mut index_pointer)?;
        let mut columns = vec![0i32; n_non_zero];
        read_i32_array(&mut file, &mut columns)?;
        let mut values = vec![0f32; n_non_zero];
        read_f32_array(&mut file, &mut values)?;

        Ok(CsrMatrix {
            num_rows: n_row,
            index_pointer,
            columns,
            values,
        })
    }

    fn row_sparse(&self, row: usize) -> Result<Vec<(u32, f32)>> {
        if row >= self.num_rows {
            bail!(
                "index {row} out of range (dataset has {} points)",
                self.num_rows
            );
        }
        let start = self.index_pointer[row] as usize;
        let end = self.index_pointer[row + 1] as usize;
        let mut pairs = Vec::with_capacity(end.saturating_sub(start));
        for j in start..end {
            pairs.push((self.columns[j] as u32, self.values[j]));
        }
        Ok(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn csr_matrix_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.csr");
        let mut file = std::fs::File::create(&path).unwrap();
        let header: [i64; 3] = [4, 4, 7];
        for value in header {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let pointers = [0i64, 2, 3, 5, 7];
        for value in pointers {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let columns = [0i32, 2, 2, 1, 3, 0, 2];
        for value in columns {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let values = [1f32, 3.0, 2.0, 3.0, 6.0, 4.0, 5.0];
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }

        let matrix = CsrMatrix::open(&path).unwrap();
        assert_eq!(matrix.row_sparse(0).unwrap(), vec![(0, 1.0), (2, 3.0)]);
        assert_eq!(matrix.row_sparse(3).unwrap(), vec![(0, 4.0), (2, 5.0)]);
    }
}
