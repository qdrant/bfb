//! Reads a ColBERT-style multivector dataset: a directory holding
//! `vectors.npy` (a flat `[total_subvectors, dim]` float array, in the same
//! format as the plain `npy` dataset) and `offsets.npy` (a 1-D int array of
//! length `num_points + 1` giving row boundaries into `vectors.npy`).
//!
//! An optional `queries/` sub-directory holds a query set in the same ragged
//! form, plus `neighbors.npy` (`[num_queries, k]` ids into the corpus) giving
//! the ground truth each query is scored against. It may also hold
//! `prefetch.npy`, a `[num_queries, dim]` array of the same queries in a
//! single-vector form, for a two-stage query whose first stage searches a
//! different vector.
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
    corpus: Ragged,
    /// Present when the dataset ships a `queries/` directory.
    queries: Option<QuerySet>,
}

/// A `vectors.npy` + `offsets.npy` pair: the ragged unit both the corpus and a
/// query set are stored in.
struct Ragged {
    vectors: NpyMatrix,
    /// Row boundaries into `vectors`, length `len + 1`.
    offsets: Vec<i64>,
}

struct QuerySet {
    vectors: Ragged,
    /// Row-major `[num_queries, k]` ids into the corpus.
    neighbors: Vec<i64>,
    k: usize,
    /// The same queries as single vectors, for a prefetch stage.
    prefetch: Option<NpyMatrix>,
}

impl Ragged {
    fn open(dir: &Path) -> Result<Self> {
        let vectors = NpyMatrix::open(&dir.join("vectors.npy"))?;
        let offsets = read_offsets_npy(&dir.join("offsets.npy"))?;

        if offsets.len() < 2 {
            bail!(
                "{}/offsets.npy must have at least 2 entries (rows + 1), got {}",
                dir.display(),
                offsets.len()
            );
        }
        let last = *offsets.last().unwrap();
        if last < 0 || last as usize != vectors.rows() {
            bail!(
                "{}/offsets.npy's last entry ({last}) does not match vectors.npy's row count ({})",
                dir.display(),
                vectors.rows()
            );
        }

        Ok(Ragged { vectors, offsets })
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn row_at(&self, idx: usize) -> Result<Vec<Vec<f32>>> {
        if idx + 1 >= self.offsets.len() {
            bail!("index {idx} out of range ({} rows)", self.len());
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

impl QuerySet {
    fn open(dir: &Path) -> Result<Self> {
        let vectors = Ragged::open(dir)?;
        let (neighbors, shape) = read_int_npy(&dir.join("neighbors.npy"))?;
        let [num_queries, k] = shape[..] else {
            bail!("queries/neighbors.npy must be 2-D [num_queries, k], got shape {shape:?}");
        };
        if num_queries != vectors.len() {
            bail!(
                "queries/neighbors.npy has {num_queries} rows but the query set has {}",
                vectors.len()
            );
        }
        let prefetch_path = dir.join("prefetch.npy");
        let prefetch = if prefetch_path.is_file() {
            let matrix = NpyMatrix::open(&prefetch_path)?;
            if matrix.rows() != vectors.len() {
                bail!(
                    "queries/prefetch.npy has {} rows but the query set has {}",
                    matrix.rows(),
                    vectors.len()
                );
            }
            Some(matrix)
        } else {
            None
        };

        Ok(QuerySet {
            vectors,
            neighbors,
            k,
            prefetch,
        })
    }
}

impl MultivectorReader {
    pub fn open(path: &Path) -> Result<Self> {
        let corpus = Ragged::open(path)?;
        let query_dir = path.join("queries");
        let queries = if query_dir.is_dir() {
            Some(QuerySet::open(&query_dir)?)
        } else {
            None
        };
        Ok(MultivectorReader { corpus, queries })
    }

    pub fn num_points(&self) -> usize {
        self.corpus.len()
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<Vec<f32>>> {
        self.corpus.row_at(idx)
    }

    /// Queries in the dataset's query set, 0 when it ships none.
    pub fn num_queries(&self) -> usize {
        self.queries.as_ref().map_or(0, |q| q.vectors.len())
    }

    pub fn query_at(&self, idx: usize) -> Result<Vec<Vec<f32>>> {
        self.query_set()?.vectors.row_at(idx)
    }

    /// Ground-truth ids for a query, as ids into the corpus.
    pub fn neighbors_at(&self, idx: usize) -> Result<Vec<u64>> {
        let queries = self.query_set()?;
        if idx >= queries.vectors.len() {
            bail!(
                "query {idx} out of range ({} queries)",
                queries.vectors.len()
            );
        }
        queries.neighbors[idx * queries.k..(idx + 1) * queries.k]
            .iter()
            .map(|&id| u64::try_from(id).context("negative id in queries/neighbors.npy"))
            .collect()
    }

    /// Whether the query set carries single-vector queries for a prefetch stage.
    pub fn has_prefetch_queries(&self) -> bool {
        self.queries.as_ref().is_some_and(|q| q.prefetch.is_some())
    }

    /// A query's single-vector form, for the first stage of a two-stage query.
    pub fn prefetch_query_at(&self, idx: usize) -> Result<Vec<f32>> {
        self.query_set()?
            .prefetch
            .as_ref()
            .context("query set ships no prefetch.npy")?
            .row(idx)
    }

    fn query_set(&self) -> Result<&QuerySet> {
        self.queries
            .as_ref()
            .context("multivector dataset ships no queries/ directory")
    }
}

fn read_offsets_npy(path: &Path) -> Result<Vec<i64>> {
    let (values, shape) = read_int_npy(path)?;
    if shape.len() != 1 {
        bail!("expected a 1-D .npy array for offsets, got shape {shape:?}");
    }
    Ok(values)
}

/// Minimal parser for a numeric `.npy` array (`int32`/`int64`, signed or
/// unsigned, or `float32`/`float64` downcast to `i64`), returning the values in
/// C order with the shape beside them. Used for the 1-D `offsets.npy` and the
/// 2-D `queries/neighbors.npy`. Read in full rather than mmapped: both are tiny
/// next to `vectors.npy`.
fn read_int_npy(path: &Path) -> Result<(Vec<i64>, Vec<usize>)> {
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
            "unsupported dtype {other:?} in {} (expected int32/int64/uint32/uint64/float32/float64)",
            path.display()
        ),
    };
    let elem_size = elem.size();

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        bail!("{} is Fortran-ordered; expected C order", path.display());
    }

    let shape = extract_shape(header)?;
    let n: usize = shape.iter().product();
    let data = &buf[header_end..];
    if data.len() < n * elem_size {
        bail!(
            "{} is truncated: need {} bytes of data, got {}",
            path.display(),
            n * elem_size,
            data.len()
        );
    }

    let values = (0..n)
        .map(|i| elem.read(&data[i * elem_size..(i + 1) * elem_size]))
        .collect();
    Ok((values, shape))
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

/// Extract the shape tuple from a `.npy` header dict.
fn extract_shape(header: &str) -> Result<Vec<usize>> {
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
    Ok(dims)
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
    fn reads_query_set_with_ground_truth_and_prefetch() {
        use crate::dataset::fixtures::write_multivector_dataset;

        let dir = tempfile::tempdir().unwrap();
        // 3 corpus points, 2 queries of 2 and 1 sub-vectors, top-2 ground truth.
        write_multivector_dataset(
            dir.path(),
            &[0, 2, 2, 5],
            &[0, 2, 3],
            &[0, 2, 1, 2],
            2,
            3,
            Some(4),
        );

        let reader = MultivectorReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 3);
        assert_eq!(reader.num_queries(), 2);
        assert_eq!(reader.query_at(0).unwrap().len(), 2);
        assert_eq!(reader.query_at(1).unwrap().len(), 1);
        assert_eq!(reader.neighbors_at(0).unwrap(), vec![0, 2]);
        assert_eq!(reader.neighbors_at(1).unwrap(), vec![1, 2]);
        assert!(reader.has_prefetch_queries());
        assert_eq!(reader.prefetch_query_at(1).unwrap().len(), 4);
    }

    /// A corpus with no `queries/` directory is still a valid dataset to upload.
    #[test]
    fn corpus_without_a_query_set_has_no_queries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vectors.npy"), make_ramp_npy(0, 5, 3)).unwrap();
        std::fs::write(
            dir.path().join("offsets.npy"),
            make_offsets_npy(&[0, 2, 2, 5]),
        )
        .unwrap();

        let reader = MultivectorReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_queries(), 0);
        assert!(!reader.has_prefetch_queries());
        assert!(reader.query_at(0).is_err());
        assert!(reader.neighbors_at(0).is_err());
    }

    /// Ground truth that does not line up with the queries would score each
    /// search against another query's answers.
    #[test]
    fn rejects_neighbors_that_do_not_match_the_query_count() {
        use crate::dataset::fixtures::write_multivector_dataset;

        let dir = tempfile::tempdir().unwrap();
        write_multivector_dataset(dir.path(), &[0, 2, 2, 5], &[0, 2, 3], &[0, 2], 2, 3, None);

        let err = match MultivectorReader::open(dir.path()) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("opened a dataset whose ground truth does not match its queries"),
        };
        assert!(err.contains("1 rows but the query set has 2"), "{err}");
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
