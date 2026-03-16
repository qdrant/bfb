use rand::SeedableRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal};
use std::fmt;

/// Generator for structured dense vectors using Poincaré disk sampling
/// combined with low-rank random projection and Gaussian noise.
///
/// This produces vectors with inherent cluster structure that indexes
/// well with HNSW, unlike uniform random vectors which suffer from
/// the curse of dimensionality.
pub struct StructuredVectorGenerator {
    /// Projection matrix of shape [intrinsic_dim × target_dim], stored row-major.
    projection: Vec<f32>,
    /// Intrinsic dimension (Poincaré disk dimension).
    intrinsic_dim: usize,
    /// Target dimension (output vector dimension).
    target_dim: usize,
    /// Gaussian noise standard deviation.
    noise_sigma: f32,
    /// Whether to L2-normalize the output vectors.
    normalize: bool,
}

impl fmt::Debug for StructuredVectorGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuredVectorGenerator")
            .field("intrinsic_dim", &self.intrinsic_dim)
            .field("target_dim", &self.target_dim)
            .field("noise_sigma", &self.noise_sigma)
            .field("normalize", &self.normalize)
            .finish()
    }
}

impl StructuredVectorGenerator {
    pub fn new(
        intrinsic_dim: usize,
        target_dim: usize,
        noise_sigma: f32,
        normalize: bool,
        seed: u64,
    ) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let projection: Vec<f32> = (0..intrinsic_dim * target_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Self {
            projection,
            intrinsic_dim,
            target_dim,
            noise_sigma,
            normalize,
        }
    }

    /// Generate a structured dense vector.
    pub fn generate(&self, rng: &mut impl Rng) -> Vec<f32> {
        let poincare_point = self.sample_poincare_disk(rng);

        // Project from intrinsic_dim to target_dim via matrix multiply
        let mut result = vec![0.0f32; self.target_dim];
        for (i, &coord) in poincare_point.iter().enumerate() {
            let row_offset = i * self.target_dim;
            for j in 0..self.target_dim {
                result[j] += coord * self.projection[row_offset + j];
            }
        }

        // Add Gaussian noise
        if self.noise_sigma > 0.0 {
            let noise_dist = Normal::new(0.0f32, self.noise_sigma).unwrap();
            for val in &mut result {
                *val += noise_dist.sample(rng);
            }
        }

        // Optionally L2-normalize (for cosine distance)
        if self.normalize {
            let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut result {
                    *val /= norm;
                }
            }
        }

        result
    }

    /// Sample a point from the Poincaré disk (unit ball) of dimension `intrinsic_dim`.
    ///
    /// Uses the Gaussian direction + radial scaling method:
    /// 1. Sample direction from Normal(0,1), normalize to unit sphere
    /// 2. Sample radius r = U^(1/k) for uniform distribution in the ball
    fn sample_poincare_disk(&self, rng: &mut impl Rng) -> Vec<f32> {
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let mut point: Vec<f32> = (0..self.intrinsic_dim)
            .map(|_| normal.sample(rng))
            .collect();

        // Normalize to unit sphere
        let norm: f32 = point.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut point {
                *val /= norm;
            }
        }

        // Radial scaling: U^(1/k) gives uniform distribution in the k-dimensional ball
        let u: f32 = rng.random_range(0.0..1.0);
        let radius = u.powf(1.0 / self.intrinsic_dim as f32);
        for val in &mut point {
            *val *= radius;
        }

        point
    }
}
