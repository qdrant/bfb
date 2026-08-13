use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use super::jsonl::JsonlStore;
use super::npy::NpyMatrix;

/// An extracted `.tgz` bundle: `vectors.npy` plus optional `payloads.jsonl` and
/// `tests.jsonl` (ann-filtering-benchmark-datasets layout).
pub struct TarReader {
    vectors: NpyMatrix,
    payloads: Option<JsonlStore>,
    /// Query set + ground truth (`tests.jsonl`), if present. Each line has the
    /// shape `{ "query": [..], "conditions": {..}, "closest_ids": [..],
    /// "closest_scores": [..] }` (ann-filtering-benchmark-datasets format).
    queries: Option<JsonlStore>,
}

impl TarReader {
    pub fn open(path: &Path) -> Result<Self> {
        let vectors = NpyMatrix::open(&path.join("vectors.npy"))?;

        let payloads_path = path.join("payloads.jsonl");
        let payloads = if payloads_path.exists() {
            Some(JsonlStore::open(&payloads_path)?)
        } else {
            None
        };

        let queries_path = path.join("tests.jsonl");
        let queries = if queries_path.exists() {
            Some(JsonlStore::open(&queries_path)?)
        } else {
            None
        };

        Ok(TarReader {
            vectors,
            payloads,
            queries,
        })
    }

    pub fn num_points(&self) -> usize {
        self.vectors.rows()
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        self.vectors.row(idx)
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

    /// Number of queries in `tests.jsonl` (0 if absent).
    pub fn num_queries(&self) -> usize {
        self.queries.as_ref().map_or(0, |store| store.len())
    }

    fn query_line(&self, idx: usize) -> Result<Value> {
        let store = self
            .queries
            .as_ref()
            .context("dataset has no tests.jsonl (no query set)")?;
        store
            .value_at(idx)?
            .with_context(|| format!("query index {idx} out of range"))
    }

    /// A dense query vector from `tests.jsonl` (`query` field).
    pub fn query_at(&self, idx: usize) -> Result<Vec<f32>> {
        let line = self.query_line(idx)?;
        let query = line
            .get("query")
            .context("tests.jsonl line is missing `query`")?;
        parse_f32_array(query).context("tests.jsonl `query` is not an array of numbers")
    }

    /// The whole query set: every query vector with its ground-truth ids.
    ///
    /// Reads each row exactly once. Going through [`Self::query_at`] and
    /// [`Self::query_ground_truth`] per index would reopen the file and re-parse
    /// the same row twice — once per field — which dominated startup on a query
    /// set of 2048-d vectors.
    pub fn read_query_set(&self) -> Result<(Vec<Vec<f32>>, Vec<Vec<u64>>)> {
        let store = self
            .queries
            .as_ref()
            .context("dataset has no tests.jsonl (no query set)")?;
        let rows: Vec<QueryRow> = store.deserialize_all()?;
        Ok(rows
            .into_iter()
            .map(|row| (row.query, row.closest_ids))
            .unzip())
    }

    /// Ground-truth nearest-neighbor ids for a query (`closest_ids` field).
    pub fn query_ground_truth(&self, idx: usize) -> Result<Vec<u64>> {
        let line = self.query_line(idx)?;
        let ids = line
            .get("closest_ids")
            .context("tests.jsonl line is missing `closest_ids`")?;
        ids.as_array()
            .context("tests.jsonl `closest_ids` is not an array")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .context("tests.jsonl `closest_ids` element is not a non-negative integer")
            })
            .collect()
    }
}

/// The fields of a `tests.jsonl` row that a benchmark run needs. `conditions`
/// and `closest_scores` are deliberately absent so serde skips them.
#[derive(serde::Deserialize)]
struct QueryRow {
    query: Vec<f32>,
    closest_ids: Vec<u64>,
}

fn parse_f32_array(value: &Value) -> Option<Vec<f32>> {
    value
        .as_array()?
        .iter()
        .map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Option<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::fixtures::make_ramp_npy;

    fn write_dataset(dir: &Path) {
        std::fs::write(dir.join("vectors.npy"), make_ramp_npy(0, 2, 3)).unwrap();
    }

    #[test]
    fn reads_vectors() {
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path());

        let reader = TarReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_points(), 2);
        assert_eq!(reader.vector_at(1).unwrap(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn reads_queries_and_ground_truth() {
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path());
        std::fs::write(
            dir.path().join("tests.jsonl"),
            "{\"query\": [1.0, 2.0, 3.0], \"conditions\": {}, \"closest_ids\": [1, 0], \"closest_scores\": [0.9, 0.8]}\n\
             {\"query\": [4.0, 5.0, 6.0], \"conditions\": {}, \"closest_ids\": [0], \"closest_scores\": [0.7]}\n",
        )
        .unwrap();

        let reader = TarReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_queries(), 2);
        assert_eq!(reader.query_at(0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(reader.query_ground_truth(0).unwrap(), vec![1, 0]);
        assert_eq!(reader.query_ground_truth(1).unwrap(), vec![0]);

        // The bulk pass must agree with the indexed reads it replaces.
        let (vectors, ground_truth) = reader.read_query_set().unwrap();
        assert_eq!(vectors, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(ground_truth, vec![vec![1, 0], vec![0]]);
    }

    #[test]
    fn no_tests_jsonl_means_no_queries() {
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path());

        let reader = TarReader::open(dir.path()).unwrap();
        assert_eq!(reader.num_queries(), 0);
        assert!(reader.query_at(0).is_err());
    }
}
