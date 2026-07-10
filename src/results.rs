//! The unified benchmark results document written by `--json`.
//!
//! One typed `{config, results}` object covering *every* phase of a run —
//! upload, index wait, search, scroll — so consumers read BFB's own output
//! instead of scraping stdout or timing phases from the shell.

use std::fs::File;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::args::Args;
use crate::processor::Timing;

/// Top-level document: `{ config: {...}, results: {...} }`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResults {
    pub config: RunConfig,
    pub results: PhaseResults,
}

/// Echo of the run's parameters, so a results file is self-describing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunConfig {
    pub bfb_version: String,
    pub collection_name: String,
    pub num_vectors: usize,
    pub batch_size: usize,
    pub parallel: usize,
    pub threads: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rps: Option<f64>,
    /// Path of the YAML config driving `bfb upload` / `bfb search`, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_file: Option<String>,
}

/// One entry per phase. A phase that did not run is omitted.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PhaseResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload: Option<UploadPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<IndexPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<QueryPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scroll: Option<QueryPhase>,
}

/// M2: upload wall time and throughput.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UploadPhase {
    pub duration_secs: f64,
    /// Points actually sent (may be < requested if the run was interrupted).
    pub num_points: usize,
    pub points_per_sec: f64,
}

/// M2: time spent waiting for the collection to report a green index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexPhase {
    pub wait_secs: f64,
}

/// A search or scroll phase: latency/throughput series plus their summaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QueryPhase {
    pub duration_secs: f64,
    /// Per-request latency reported by the server, in seconds.
    pub server_timings: Vec<f32>,
    /// End-to-end per-request latency, in seconds.
    pub full_timings: Vec<f32>,
    pub rps: Vec<f32>,
    pub qps: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_time: Option<Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_time: Option<Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<PrecisionSummary>,
}

/// Distribution of a timing series. Percentiles use nearest-rank on sorted data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub min: f32,
    pub avg: f64,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PrecisionSummary {
    pub avg: f32,
    pub p50: f32,
}

impl Summary {
    /// Summarize `values`, which **must already be sorted** by `value` ascending.
    /// Returns `None` for an empty series.
    pub fn from_sorted(values: &[Timing]) -> Option<Self> {
        let first = values.first()?;
        Some(Summary {
            min: first.value,
            avg: values.iter().map(|x| x.value as f64).sum::<f64>() / values.len() as f64,
            p50: percentile(values, 0.50),
            p95: percentile(values, 0.95),
            max: values.last().expect("non-empty").value,
        })
    }
}

/// Nearest-rank percentile of a sorted, non-empty series.
fn percentile(sorted: &[Timing], q: f32) -> f32 {
    let index = ((sorted.len() as f32 * q) as usize).min(sorted.len() - 1);
    sorted[index].value
}

impl PrecisionSummary {
    /// Summarize precision/recall samples. Sorts a copy; returns `None` if empty.
    pub fn new(mut precisions: Vec<f32>) -> Option<Self> {
        if precisions.is_empty() {
            return None;
        }
        let avg = precisions.iter().sum::<f32>() / precisions.len() as f32;
        precisions.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((precisions.len() as f32 * 0.50) as usize).min(precisions.len() - 1);
        Some(PrecisionSummary {
            avg,
            p50: precisions[index],
        })
    }
}

impl UploadPhase {
    pub fn new(duration_secs: f64, num_points: usize) -> Self {
        UploadPhase {
            duration_secs,
            num_points,
            points_per_sec: if duration_secs > 0.0 {
                num_points as f64 / duration_secs
            } else {
                0.0
            },
        }
    }
}

impl BenchmarkResults {
    pub fn new(args: &Args, config_file: Option<String>) -> Self {
        BenchmarkResults {
            config: RunConfig {
                bfb_version: env!("CARGO_PKG_VERSION").to_string(),
                collection_name: args.collection_name.clone(),
                num_vectors: args.num_vectors_or_default(),
                batch_size: args.batch_size,
                parallel: args.parallel,
                threads: args.threads,
                rps: args.rps,
                config_file,
            },
            results: PhaseResults::default(),
        }
    }

    /// Write the document to `--json <path>`, if the flag was given.
    pub fn write_if_requested(&self, args: &Args) -> Result<()> {
        let Some(path) = args.json.as_ref() else {
            return Ok(());
        };
        println!("--- Writing results to json file ---");
        let file =
            File::create(path).with_context(|| format!("failed to create results file {path}"))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write results to {path}"))?;
        println!("Results written to {path}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timings(values: &[f32]) -> Vec<Timing> {
        values
            .iter()
            .map(|&value| Timing {
                delay_millis: 0,
                value,
            })
            .collect()
    }

    #[test]
    fn summary_of_empty_series_is_none() {
        assert!(Summary::from_sorted(&[]).is_none());
        assert!(PrecisionSummary::new(vec![]).is_none());
    }

    #[test]
    fn summary_reports_distribution() {
        let s = Summary::from_sorted(&timings(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 4.0);
        assert_eq!(s.avg, 2.5);
        assert_eq!(s.p50, 3.0); // nearest-rank: index 2
    }

    #[test]
    fn summary_of_single_sample_never_indexes_past_the_end() {
        let s = Summary::from_sorted(&timings(&[7.0])).unwrap();
        assert_eq!((s.min, s.p50, s.p95, s.max), (7.0, 7.0, 7.0, 7.0));
    }

    #[test]
    fn precision_summary_sorts_before_taking_median() {
        let p = PrecisionSummary::new(vec![0.9, 0.1, 0.5]).unwrap();
        assert_eq!(p.p50, 0.5);
        assert!((p.avg - 0.5).abs() < 1e-6);
    }

    #[test]
    fn upload_phase_derives_throughput() {
        let u = UploadPhase::new(2.0, 1000);
        assert_eq!(u.points_per_sec, 500.0);
        // A zero-length run must not produce inf/NaN.
        assert_eq!(UploadPhase::new(0.0, 10).points_per_sec, 0.0);
    }

    #[test]
    fn skipped_phases_are_omitted_from_json() {
        let results = PhaseResults {
            index: Some(IndexPhase { wait_secs: 1.5 }),
            ..Default::default()
        };
        let json = serde_json::to_string(&results).unwrap();
        assert_eq!(json, r#"{"index":{"wait_secs":1.5}}"#);
    }

    #[test]
    fn document_roundtrips() {
        let doc = BenchmarkResults {
            config: RunConfig {
                bfb_version: "0.1.1".into(),
                collection_name: "benchmark".into(),
                num_vectors: 100,
                batch_size: 8,
                parallel: 2,
                threads: 4,
                rps: None,
                config_file: Some("c.yaml".into()),
            },
            results: PhaseResults {
                upload: Some(UploadPhase::new(1.0, 100)),
                index: Some(IndexPhase { wait_secs: 0.5 }),
                search: Some(QueryPhase {
                    duration_secs: 2.0,
                    server_timings: vec![1.0],
                    full_timings: vec![2.0],
                    rps: vec![50.0],
                    qps: vec![50.0],
                    server_time: Summary::from_sorted(&timings(&[1.0])),
                    request_time: None,
                    precision: PrecisionSummary::new(vec![1.0]),
                }),
                scroll: None,
            },
        };

        let json = serde_json::to_string(&doc).unwrap();
        assert_eq!(
            serde_json::from_str::<BenchmarkResults>(&json).unwrap(),
            doc
        );
    }
}
