//! Per-stage indexing timings from `GET /collections/{c}/optimizations`.
//!
//! This is the one thing BFB cannot get over gRPC — the public API exposes it
//! only over REST (the gRPC `GetShardOptimizations` is an internal, node-to-node
//! RPC that ships this same JSON as opaque bytes). So we speak HTTP here, on the
//! REST port, reusing the `ureq` client already pulled in for dataset downloads.
//!
//! The server keeps the last 16 completed optimizations in a ring buffer, so
//! this must be read *after* indexing finishes but before too many further
//! optimizations run.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Server default for `?completed_limit=`; also the depth of the server's
/// completed-optimization ring buffer.
const DEFAULT_COMPLETED_LIMIT: usize = 16;

/// Bound the REST call when `--timeout` is unset; `ureq` applies none by default.
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

// ------------------------- wire types (server shape) ----------------------

/// The REST envelope: `{result, status, time}`.
#[derive(Debug, Deserialize)]
struct RestEnvelope<T> {
    result: T,
}

#[derive(Debug, Default, Deserialize)]
pub struct OptimizationsResponse {
    #[serde(default)]
    pub running: Vec<Optimization>,
    #[serde(default)]
    pub completed: Option<Vec<Optimization>>,
}

#[derive(Debug, Deserialize)]
pub struct Optimization {
    /// Identifies this optimization run. Used to tell ours apart from those
    /// that had already completed before the phase started.
    pub uuid: String,
    /// Name of the optimizer that ran (e.g. `indexing`, `merge`).
    pub optimizer: String,
    /// `"done"`, or a single-key object for `cancelled` / `error` (each carrying
    /// a reason string), or `"optimizing"` while still running.
    pub status: serde_json::Value,
    pub progress: ProgressTree,
}

impl Optimization {
    /// Only `done` optimizations have trustworthy stage timings; a cancelled or
    /// errored one stopped partway through its tree.
    fn is_done(&self) -> bool {
        self.status.as_str() == Some("done")
    }
}

/// A recursive stage tree. Each node carries its own wall-clock duration.
#[derive(Debug, Deserialize)]
pub struct ProgressTree {
    pub name: String,
    #[serde(default)]
    pub duration_sec: Option<f64>,
    #[serde(default)]
    pub children: Vec<ProgressTree>,
}

// ------------------------- reported types ---------------------------------

/// Per-stage aggregate across every completed optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StageAggregate {
    /// How many optimizations reported this stage.
    pub count: usize,
    pub total_secs: f64,
    pub max_secs: f64,
}

/// Per-stage breakdown of the optimizations the server ran, attached to the
/// index phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OptimizationsReport {
    /// Number of `done` optimizations aggregated below.
    pub completed: usize,
    /// Optimizations that had already completed before this phase started, and
    /// so were excluded. See [`Baseline`].
    pub pre_existing_skipped: usize,
    /// The server returned a full page of completed optimizations, so older
    /// ones may have been evicted from its ring buffer before we read it —
    /// meaning `stages` can undercount. Run fewer optimizations per phase.
    pub possibly_truncated: bool,
    /// Completed-but-not-`done` entries (cancelled or errored), whose partial
    /// stage timings are excluded from `stages`.
    pub unfinished: usize,
    /// Number of optimizations still running when we read the endpoint.
    pub running: usize,
    /// Wall-clock seconds summed over every completed optimization's root node.
    pub total_secs: f64,
    /// Which optimizers ran, e.g. `indexing` → 5, `merge` → 1.
    pub optimizers: BTreeMap<String, usize>,
    /// Stage path → aggregate, e.g. `vector_index/main_graph`, `quantization`,
    /// `payload_index/keyword:color`.
    pub stages: BTreeMap<String, StageAggregate>,
}

impl OptimizationsReport {
    fn from_response(response: &OptimizationsResponse, baseline: &Baseline) -> Self {
        let completed = response.completed.as_deref().unwrap_or_default();

        let mut report = OptimizationsReport {
            running: response.running.len(),
            // The server caps the page, and its own buffer, at this many.
            possibly_truncated: completed.len() >= DEFAULT_COMPLETED_LIMIT,
            ..Default::default()
        };

        for optimization in completed {
            // Optimizations that had already finished before the phase began
            // belong to an earlier phase, not this one.
            if baseline.contains(&optimization.uuid) {
                report.pre_existing_skipped += 1;
                continue;
            }
            if !optimization.is_done() {
                report.unfinished += 1;
                continue;
            }
            report.completed += 1;
            *report
                .optimizers
                .entry(optimization.optimizer.clone())
                .or_default() += 1;
            report.total_secs += optimization.progress.duration_sec.unwrap_or(0.0);
            // The root node ("Segment Optimizing") is the total, already counted
            // above; only its descendants are stages.
            for child in &optimization.progress.children {
                report.collect_stage(child, "");
            }
        }

        report
    }

    /// Fold `node` and its descendants into `stages`, keyed by path.
    ///
    /// Keys are paths rather than bare names because a name can repeat under
    /// different parents with entirely different meanings: a payload field shows
    /// up once under `payload_index` (building the field index) and again under
    /// `additional_links` (adding HNSW links for it). Collapsing those into one
    /// `keyword:color` bucket would sum two unrelated costs.
    ///
    /// Under `vector_index` the next level is the *vector name*, which is empty
    /// for the unnamed default vector. Such levels are skipped rather than
    /// recorded, so a default vector yields `vector_index/main_graph` (not
    /// `vector_index//main_graph`) while a named one yields
    /// `vector_index/image/main_graph`.
    fn collect_stage(&mut self, node: &ProgressTree, parent_path: &str) {
        let path = if node.name.is_empty() {
            parent_path.to_string()
        } else if parent_path.is_empty() {
            node.name.clone()
        } else {
            format!("{parent_path}/{}", node.name)
        };

        if !node.name.is_empty() {
            let secs = node.duration_sec.unwrap_or(0.0);
            let stage = self.stages.entry(path.clone()).or_default();
            stage.count += 1;
            stage.total_secs += secs;
            stage.max_secs = stage.max_secs.max(secs);
        }

        for child in &node.children {
            self.collect_stage(child, &path);
        }
    }

    /// Print the stages slowest-first.
    pub fn print(&self) {
        if self.completed == 0 {
            println!("No completed optimizations reported by the server");
            return;
        }
        println!(
            "--- Optimization stages ({} completed, {:.3} s total) ---",
            self.completed, self.total_secs
        );
        if self.unfinished > 0 {
            println!(
                "({} cancelled/errored optimization(s) excluded)",
                self.unfinished
            );
        }
        if self.possibly_truncated {
            println!(
                "(warning: the server's completed-optimization buffer was full; \
                 older stages may be missing)"
            );
        }
        let mut stages: Vec<_> = self.stages.iter().collect();
        stages.sort_unstable_by(|(_, a), (_, b)| b.total_secs.total_cmp(&a.total_secs));
        for (name, stage) in stages {
            println!(
                "{name}: {:.6} s total over {} run(s), max {:.6} s",
                stage.total_secs, stage.count, stage.max_secs
            );
        }
    }
}

// ------------------------- fetching ---------------------------------------

/// Derive the REST base URL from a gRPC `--uri`.
///
/// Qdrant serves gRPC on 6334 and REST on 6333, so this is a one-port shift.
/// A `--uri` on any other port is returned unchanged.
pub fn rest_url_from_grpc(uri: &str) -> String {
    let trimmed = uri.trim_end_matches('/');
    match trimmed.rsplit_once(":6334") {
        Some((head, "")) => format!("{head}:6333"),
        _ => trimmed.to_string(),
    }
}

/// The optimizations that had already completed when a phase began.
///
/// The server reports a rolling window of completed optimizations, not the ones
/// belonging to any particular phase. Without a baseline, `bfb wait-index` would
/// happily attribute a preceding `bfb create-index`'s work to indexing.
#[derive(Debug, Default, Clone)]
pub struct Baseline(HashSet<String>);

impl Baseline {
    fn contains(&self, uuid: &str) -> bool {
        self.0.contains(uuid)
    }

    /// An empty baseline: every completed optimization counts.
    pub fn none() -> Self {
        Baseline::default()
    }
}

/// Record which optimizations have already completed, to exclude them later.
pub fn baseline(
    rest_url: &str,
    collection: &str,
    api_key: Option<&str>,
    timeout: Option<Duration>,
) -> Result<Baseline> {
    let response = get(rest_url, collection, api_key, timeout)?;
    Ok(Baseline(
        response
            .completed
            .unwrap_or_default()
            .into_iter()
            .map(|optimization| optimization.uuid)
            .collect(),
    ))
}

/// Read completed optimization stages for `collection`, excluding anything
/// already present in `baseline`.
pub fn fetch(
    rest_url: &str,
    collection: &str,
    api_key: Option<&str>,
    baseline: &Baseline,
    timeout: Option<Duration>,
) -> Result<OptimizationsReport> {
    let response = get(rest_url, collection, api_key, timeout)?;
    Ok(OptimizationsReport::from_response(&response, baseline))
}

fn get(
    rest_url: &str,
    collection: &str,
    api_key: Option<&str>,
    timeout: Option<Duration>,
) -> Result<OptimizationsResponse> {
    let url = format!(
        "{}/collections/{collection}/optimizations?with=completed&completed_limit={DEFAULT_COMPLETED_LIMIT}",
        rest_url.trim_end_matches('/')
    );

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(timeout.unwrap_or(DEFAULT_HTTP_TIMEOUT)))
            .build(),
    );
    let mut request = agent.get(&url);
    if let Some(key) = api_key {
        request = request.header("api-key", key);
    }

    let body = request
        .call()
        .with_context(|| format!("failed to GET {url}"))?
        .into_body()
        .read_to_string()
        .with_context(|| format!("failed to read response from {url}"))?;

    let envelope: RestEnvelope<OptimizationsResponse> = serde_json::from_str(&body)
        .with_context(|| format!("failed to parse optimizations response from {url}"))?;
    Ok(envelope.result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_fixture() -> OptimizationsResponse {
        let text = include_str!("../tests/fixtures/optimizations_completed.json");
        let envelope: RestEnvelope<OptimizationsResponse> = serde_json::from_str(text).unwrap();
        envelope.result
    }

    #[test]
    fn parses_real_server_response() {
        let response = parse_fixture();
        assert_eq!(response.completed.as_ref().unwrap().len(), 2);
        assert!(response.running.is_empty());
        assert_eq!(
            response.completed.as_ref().unwrap()[0].optimizer,
            "indexing"
        );
    }

    #[test]
    fn aggregates_the_stages_the_benchmarks_ask_for() {
        let report = OptimizationsReport::from_response(&parse_fixture(), &Baseline::none());

        assert_eq!(report.completed, 2);
        assert_eq!(report.running, 0);

        // The stages the indexing benchmarks are written against.
        for stage in ["vector_index/main_graph", "quantization", "payload_index"] {
            assert!(
                report.stages.contains_key(stage),
                "missing stage {stage}; got {:?}",
                report.stages.keys().collect::<Vec<_>>()
            );
        }

        // The fixture uses the unnamed default vector, so the vector-name level
        // under `vector_index` is empty and must collapse out of the path.
        let main_graph = &report.stages["vector_index/main_graph"];
        assert_eq!(main_graph.count, 2, "one per completed optimization");
        assert!(main_graph.total_secs > 0.0);
        assert!(main_graph.max_secs <= main_graph.total_secs);

        // The unnamed wrapper node must not become a stage, nor leave an empty
        // path segment behind.
        assert!(!report.stages.contains_key(""));
        assert!(report.stages.keys().all(|k| !k.contains("//")));

        // The same field name appears under two different parents with two
        // different meanings; they must not be summed into one bucket.
        assert!(report.stages.contains_key("payload_index/keyword:a"));
        assert!(
            report
                .stages
                .contains_key("vector_index/additional_links/keyword:a")
        );
    }

    #[test]
    fn total_secs_sums_root_nodes_only() {
        let report = OptimizationsReport::from_response(&parse_fixture(), &Baseline::none());
        // Root duration exceeds any single child stage, and is not the sum of
        // all stages (which would double-count nested nodes).
        assert!(report.total_secs > report.stages["vector_index/main_graph"].total_secs);
        let stage_sum: f64 = report.stages.values().map(|s| s.total_secs).sum();
        assert!(
            stage_sum > report.total_secs,
            "nested stages overlap in time"
        );
    }

    #[test]
    fn records_which_optimizers_ran() {
        let report = OptimizationsReport::from_response(&parse_fixture(), &Baseline::none());
        assert_eq!(report.optimizers.get("indexing"), Some(&2));
        assert_eq!(report.unfinished, 0);
    }

    #[test]
    fn cancelled_and_errored_optimizations_are_excluded() {
        // A cancelled/errored entry stopped partway through its stage tree, so
        // folding its durations in would understate the real stage cost.
        let json = r#"{
          "running": [],
          "completed": [
            {"uuid": "a", "optimizer": "indexing", "status": "done",
             "progress": {"name": "Segment Optimizing", "duration_sec": 2.0,
                          "children": [{"name": "main_graph", "duration_sec": 1.5}]}},
            {"uuid": "b", "optimizer": "indexing", "status": {"cancelled": "shutdown"},
             "progress": {"name": "Segment Optimizing", "duration_sec": 0.3,
                          "children": [{"name": "main_graph", "duration_sec": 0.1}]}},
            {"uuid": "c", "optimizer": "merge", "status": {"error": "disk full"},
             "progress": {"name": "Segment Optimizing", "duration_sec": 0.2, "children": []}}
          ]
        }"#;
        let response: OptimizationsResponse = serde_json::from_str(json).unwrap();
        let report = OptimizationsReport::from_response(&response, &Baseline::none());

        assert_eq!(report.completed, 1);
        assert_eq!(report.unfinished, 2);
        assert_eq!(report.total_secs, 2.0, "only the `done` root counts");
        assert_eq!(report.stages["main_graph"].total_secs, 1.5);
        assert_eq!(report.stages["main_graph"].count, 1);
        assert_eq!(report.optimizers.get("merge"), None);
    }

    #[test]
    fn optimizations_completed_before_the_phase_are_excluded() {
        // A `bfb create-index` run leaves completed optimizations behind; the
        // `bfb wait-index` that follows must not claim them as indexing work.
        let response = parse_fixture();
        let first = response.completed.as_ref().unwrap()[0].uuid.clone();
        let baseline = Baseline(HashSet::from([first]));

        let report = OptimizationsReport::from_response(&response, &baseline);

        assert_eq!(report.completed, 1, "only the new optimization counts");
        assert_eq!(report.pre_existing_skipped, 1);
        assert_eq!(report.stages["vector_index/main_graph"].count, 1);

        // Without the baseline, both would be attributed to this phase.
        let unfiltered = OptimizationsReport::from_response(&response, &Baseline::none());
        assert_eq!(unfiltered.completed, 2);
        assert!(unfiltered.total_secs > report.total_secs);
    }

    #[test]
    fn a_full_page_of_completed_optimizations_flags_truncation() {
        // The server keeps only the last 16; a full page means older stages may
        // already be gone, so `stages` can undercount.
        let one = r#"{"uuid": "%", "optimizer": "indexing", "status": "done",
            "progress": {"name": "Segment Optimizing", "duration_sec": 1.0, "children": []}}"#;
        let entries: Vec<String> = (0..16).map(|i| one.replace('%', &i.to_string())).collect();
        let json = format!(r#"{{"running": [], "completed": [{}]}}"#, entries.join(","));

        let response: OptimizationsResponse = serde_json::from_str(&json).unwrap();
        let report = OptimizationsReport::from_response(&response, &Baseline::none());

        assert_eq!(report.completed, 16);
        assert!(report.possibly_truncated);
    }

    #[test]
    fn a_partial_page_is_not_flagged_as_truncated() {
        let report = OptimizationsReport::from_response(&parse_fixture(), &Baseline::none());
        assert!(!report.possibly_truncated);
    }

    #[test]
    fn empty_response_reports_nothing() {
        let report = OptimizationsReport::from_response(
            &OptimizationsResponse::default(),
            &Baseline::none(),
        );
        assert_eq!(report.completed, 0);
        assert!(report.stages.is_empty());
        assert_eq!(report.total_secs, 0.0);
    }

    #[test]
    fn derives_the_rest_url_from_the_grpc_uri() {
        assert_eq!(
            rest_url_from_grpc("http://localhost:6334"),
            "http://localhost:6334".replace("6334", "6333")
        );
        assert_eq!(
            rest_url_from_grpc("https://xyz.cloud.qdrant.io:6334/"),
            "https://xyz.cloud.qdrant.io:6333"
        );
        // A non-default port is left alone rather than silently rewritten.
        assert_eq!(rest_url_from_grpc("http://host:7000"), "http://host:7000");
        // A port that merely *contains* 6334 must not match.
        assert_eq!(rest_url_from_grpc("http://host:63340"), "http://host:63340");
    }
}
