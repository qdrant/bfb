//! Random data generation: points for upload and queries for search.
//!
//! Point generation is abstracted behind [`PointGenerator`] so the upload
//! pipeline (parallelism, RPS, progress, timings) is shared between the legacy
//! flag-driven path ([`LegacyGenerator`]) and the YAML-config path
//! ([`ConfigGenerator`]).

pub mod config;
pub mod legacy;
pub mod queries;
pub mod random;

pub use config::ConfigGenerator;
pub use legacy::LegacyGenerator;
pub use queries::ConfigSearchGenerator;

use qdrant_client::Payload;
use qdrant_client::qdrant::PointStruct;

/// Produces the per-point data (id, vectors, payload). The runtime layer owns
/// *which* numeric id to use (offset / max-id logic); the generator owns the
/// *shape*.
pub trait PointGenerator: Send + Sync {
    /// Build the point for the given numeric index.
    fn make_point(&self, idx: u64) -> PointStruct;

    /// Build just a payload (used by the `--set-payload` path).
    fn make_payload(&self) -> Payload;
}
