//! Search benchmarking: the flag-driven [`SearchProcessor`] and the
//! YAML-config-driven [`ConfigSearchProcessor`], plus shared precision/recall
//! scoring.

mod from_args;
mod from_config;

pub use from_args::SearchProcessor;
pub use from_config::ConfigSearchProcessor;

use std::collections::HashSet;

use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{BatchResult, ScoredPoint};

use crate::processor::Timing;

#[derive(Debug, Default)]
struct SearchStats {
    server_timings: Vec<Timing>,
    qps: Vec<Timing>,
    rps: Vec<Timing>,
    full_timings: Vec<Timing>,
    precisions: Vec<f32>,
}

fn compare_search_results(exact_search: &[ScoredPoint], normal_search: &[ScoredPoint]) -> f32 {
    if normal_search.is_empty() {
        return 0.0;
    }

    let exact_ids: HashSet<_> = exact_search
        .iter()
        .map(|i| PointId::from(i.id.as_ref().unwrap().point_id_options.clone().unwrap()))
        .collect();

    let normal_ids: HashSet<_> = normal_search
        .iter()
        .map(|i| PointId::from(i.id.as_ref().unwrap().point_id_options.clone().unwrap()))
        .collect();

    let true_positive: usize = normal_ids.iter().filter(|i| exact_ids.contains(i)).count();
    let false_positive: usize = normal_ids.iter().filter(|i| !exact_ids.contains(i)).count();

    true_positive as f32 / (true_positive as f32 + false_positive as f32)
}

/// Recall of a single search against reference-dataset ground truth:
/// `|returned_ids ∩ expected[:top]| / top`, where `top` is bounded by both the
/// search limit and the number of ground-truth neighbors available. Ground
/// truth ids are dataset corpus indices, which match integer point ids when the
/// dataset was uploaded with the default integer id scheme.
fn recall_against_ground_truth(
    returned: &[ScoredPoint],
    expected: &[u64],
    search_limit: usize,
) -> f32 {
    let top = search_limit.min(expected.len());
    if top == 0 {
        return 1.0;
    }

    let expected_top: HashSet<u64> = expected.iter().take(top).copied().collect();

    let found = returned
        .iter()
        .filter_map(|p| match p.id.as_ref()?.point_id_options.as_ref()? {
            PointIdOptions::Num(num) => Some(*num),
            PointIdOptions::Uuid(_) => None,
        })
        .filter(|id| expected_top.contains(id))
        .count();

    found as f32 / top as f32
}

fn compare_batch_search_results(
    exact_search: &[BatchResult],
    normal_search: &[BatchResult],
) -> Vec<f32> {
    assert_eq!(exact_search.len(), normal_search.len());
    exact_search
        .iter()
        .zip(normal_search)
        .map(|(exact, normal)| compare_search_results(&exact.result, &normal.result))
        .collect()
}

// Copy of `qdrant_client::qdrant::PointId` that implements `Eq` and `Hash`.
#[derive(Clone, PartialEq, Eq, Hash)]
enum PointId {
    Num(u64),
    Uuid(String),
}

impl From<PointIdOptions> for PointId {
    fn from(value: PointIdOptions) -> Self {
        match value {
            PointIdOptions::Num(i) => PointId::Num(i),
            PointIdOptions::Uuid(i) => PointId::Uuid(i),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::PointId as GrpcPointId;

    fn scored(ids: &[u64]) -> Vec<ScoredPoint> {
        ids.iter()
            .map(|&id| ScoredPoint {
                id: Some(GrpcPointId::from(id)),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn recall_perfect_and_partial() {
        // All top-3 found.
        let returned = scored(&[1, 2, 3]);
        assert_eq!(
            recall_against_ground_truth(&returned, &[1, 2, 3, 4], 3),
            1.0
        );

        // 2 of top-3 found.
        let returned = scored(&[1, 9, 3]);
        assert!((recall_against_ground_truth(&returned, &[1, 2, 3], 3) - 2.0 / 3.0).abs() < 1e-6);

        // Ground truth shorter than the limit ⇒ denominator is bounded by it.
        let returned = scored(&[5, 6]);
        assert_eq!(recall_against_ground_truth(&returned, &[5, 6], 10), 1.0);
    }

    #[test]
    fn recall_ignores_extra_returned_ids() {
        // Returning more than `top` correct ids never exceeds 1.0.
        let returned = scored(&[1, 2, 3, 4, 5]);
        assert_eq!(recall_against_ground_truth(&returned, &[1, 2], 2), 1.0);
    }
}
