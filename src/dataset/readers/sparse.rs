use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::binary::{read_f32_array, read_i32_array, read_i64_array};

pub struct SparseReader {
    matrix: CsrMatrix,
    /// Sparse query vectors (`queries.csr`), if present.
    queries: Option<CsrMatrix>,
    /// Ground-truth nearest-neighbor ids per query (`results.gt`), if present.
    ground_truth: Option<KnnGroundTruth>,
}

impl SparseReader {
    pub fn open(path: &Path) -> Result<Self> {
        let data_path = path.join("data.csr");
        let queries_path = path.join("queries.csr");
        let queries = if queries_path.exists() {
            Some(CsrMatrix::open(&queries_path)?)
        } else {
            None
        };
        let gt_path = path.join("results.gt");
        let ground_truth = if gt_path.exists() {
            Some(KnnGroundTruth::open(&gt_path)?)
        } else {
            None
        };
        Ok(SparseReader {
            matrix: CsrMatrix::open(&data_path)?,
            queries,
            ground_truth,
        })
    }

    pub fn num_points(&self) -> usize {
        self.matrix.num_rows
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        self.matrix.row_sparse(idx)
    }

    /// Number of sparse queries in `queries.csr` (0 if absent).
    pub fn num_queries(&self) -> usize {
        self.queries.as_ref().map_or(0, |m| m.num_rows)
    }

    /// A sparse query vector from `queries.csr`.
    pub fn query_at(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        self.queries
            .as_ref()
            .context("sparse dataset has no queries.csr (no query set)")?
            .row_sparse(idx)
    }

    /// Ground-truth nearest-neighbor ids for a query, from `results.gt`.
    pub fn query_ground_truth(&self, idx: usize) -> Result<Vec<u64>> {
        self.ground_truth
            .as_ref()
            .context("sparse dataset has no results.gt (no ground truth)")?
            .row(idx)
    }
}

/// Ground-truth top-k neighbor ids for the sparse query set, stored in the
/// `results.gt` (knn) format: a `[n, k]` header of two little-endian `u32`s,
/// followed by `n*k` `i32` ids, then `n*k` `f32` scores (scores unused here).
struct KnnGroundTruth {
    num_rows: usize,
    k: usize,
    ids: Vec<i32>,
}

impl KnnGroundTruth {
    fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)
            .context("failed to read results.gt header")?;
        let num_rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let k = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let mut ids = vec![0i32; num_rows * k];
        read_i32_array(&mut file, &mut ids)?;
        Ok(KnnGroundTruth { num_rows, k, ids })
    }

    fn row(&self, idx: usize) -> Result<Vec<u64>> {
        if idx >= self.num_rows {
            bail!(
                "index {idx} out of range (results.gt has {} rows)",
                self.num_rows
            );
        }
        let start = idx * self.k;
        Ok(self.ids[start..start + self.k]
            .iter()
            .map(|&v| v as u64)
            .collect())
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
