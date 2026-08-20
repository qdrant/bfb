/// One entry of a dataset's query set: a query vector paired with the
/// ground-truth ids used to score its recall.
///
/// Query sets are read in full at startup and returned as `Vec<QueryEntry<_>>`,
/// which keeps file I/O and JSON parsing off the timed request path. `V` is the
/// vector form the caller asked for — `Vec<f32>` for dense, `Vec<(u32, f32)>`
/// for sparse.
pub struct QueryEntry<V> {
    pub vector: V,
    pub ground_truth: Vec<u64>,
    /// Raw `conditions` object the query was answered under, when the query set
    /// carries one. Kept as JSON here and turned into a filter once at startup;
    /// the ground truth only holds for a search that applies it.
    pub conditions: Option<serde_json::Value>,
}

/// A sparse query vector as `(index, value)` pairs — the `V` of a sparse
/// [`QueryEntry`]. Named so the query-set return type stays legible.
pub type SparseVector = Vec<(u32, f32)>;
