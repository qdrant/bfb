//! Uniform / Zipf sampling over a fixed set of collection slots.

use rand::Rng;
use rand::RngExt;
use rand::prelude::Distribution as RandDistribution;
use rand_distr::Zipf;

/// How work (points or queries) is spread across collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    Uniform,
    /// Zipf with exponent 1.03 over ranks `1..=n` (same default as payload text).
    Zipf,
}

/// Pick a collection index in `0..n` according to [`Distribution`].
#[derive(Debug, Clone)]
pub struct CollectionPicker {
    n: usize,
    zipf: Option<Zipf<f64>>,
}

/// One request produced while draining pre-allocated collection budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetBatch {
    pub collection: usize,
    /// Global item offset. Each collection owns one contiguous range, even
    /// though requests from those ranges are emitted in random order.
    pub offset: usize,
    pub count: usize,
}

/// Randomly drain collection budgets in batches of at most `batch_size`.
///
/// Sampling is uniform among collections that still have budget. The desired
/// uniform/Zipf skew is already represented by `budgets`; applying it again
/// while draining would skew the workload twice.
pub fn drain_budgets(budgets: &[usize], batch_size: usize, rng: &mut impl Rng) -> Vec<BudgetBatch> {
    assert!(batch_size > 0, "batch size must be positive");

    let mut remaining = budgets.to_vec();
    let mut next_offset = Vec::with_capacity(budgets.len());
    let mut offset = 0usize;
    for &budget in budgets {
        next_offset.push(offset);
        offset += budget;
    }
    let mut active: Vec<_> = remaining
        .iter()
        .enumerate()
        .filter_map(|(i, &budget)| (budget > 0).then_some(i))
        .collect();
    let mut batches = Vec::new();

    while !active.is_empty() {
        let active_idx = rng.random_range(0..active.len());
        let collection = active[active_idx];
        let count = remaining[collection].min(batch_size);
        batches.push(BudgetBatch {
            collection,
            offset: next_offset[collection],
            count,
        });
        next_offset[collection] += count;
        remaining[collection] -= count;
        if remaining[collection] == 0 {
            active.swap_remove(active_idx);
        }
    }
    batches
}

impl CollectionPicker {
    pub fn new(n: usize, distribution: Distribution) -> anyhow::Result<Self> {
        anyhow::ensure!(n > 0, "collections-count must be > 0");
        let zipf = match distribution {
            Distribution::Uniform => None,
            Distribution::Zipf => Some(
                Zipf::new(n as f64, 1.03)
                    .map_err(|e| anyhow::anyhow!("failed to build Zipf(n={n}): {e}"))?,
            ),
        };
        Ok(Self { n, zipf })
    }

    /// Sample a collection index in `0..n`.
    pub fn pick(&self, rng: &mut impl Rng) -> usize {
        match &self.zipf {
            None => rng.random_range(0..self.n),
            // Zipf samples in `1..=n`; convert to 0-based.
            Some(zipf) => ((zipf.sample(rng) as usize).saturating_sub(1)).min(self.n - 1),
        }
    }

    /// Allocate `total` points across `n` collections according to the
    /// distribution. Returns a vec of length `n` summing to `total`.
    pub fn allocate(&self, total: usize, rng: &mut impl Rng) -> Vec<usize> {
        if self.n == 1 {
            return vec![total];
        }
        match self.zipf {
            None => {
                // Even split; remainder goes to the first slots.
                let base = total / self.n;
                let rem = total % self.n;
                (0..self.n).map(|i| base + usize::from(i < rem)).collect()
            }
            Some(_) => {
                // Monte-Carlo allocation: sample `total` ranks.
                let mut counts = vec![0usize; self.n];
                for _ in 0..total {
                    counts[self.pick(rng)] += 1;
                }
                counts
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn uniform_allocate_sums() {
        let picker = CollectionPicker::new(10, Distribution::Uniform).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let counts = picker.allocate(1000, &mut rng);
        assert_eq!(counts.len(), 10);
        assert_eq!(counts.iter().sum::<usize>(), 1000);
        assert!(counts.iter().all(|&c| c == 100));
    }

    #[test]
    fn zipf_allocate_sums_and_skews() {
        let picker = CollectionPicker::new(10, Distribution::Zipf).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let counts = picker.allocate(10_000, &mut rng);
        assert_eq!(counts.iter().sum::<usize>(), 10_000);
        // Rank 0 should get strictly more than the last rank on average.
        assert!(counts[0] > counts[9]);
    }

    #[test]
    fn draining_budgets_preserves_limits_offsets_and_totals() {
        let mut rng = StdRng::seed_from_u64(7);
        let batches = drain_budgets(&[5, 0, 7], 3, &mut rng);
        assert!(batches.iter().all(|batch| batch.count <= 3));
        let mut totals = [0usize; 3];
        let mut offsets = [Vec::new(), Vec::new(), Vec::new()];
        for batch in batches {
            totals[batch.collection] += batch.count;
            offsets[batch.collection].push(batch.offset);
        }
        assert_eq!(totals, [5, 0, 7]);
        offsets.iter_mut().for_each(|values| values.sort_unstable());
        assert_eq!(offsets[0], [0, 3]);
        assert_eq!(offsets[2], [5, 8, 11]);
    }
}
