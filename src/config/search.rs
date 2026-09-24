//! YAML search-request configuration for `bfb search --file` / `--example`.
//!
//! The config file describes only the *shape* of search requests (which vectors
//! to query, optional payload filters). The *how* of searching (number of
//! queries, batch size, threads, parallelism, uri, …) stays on the CLI.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::{
    DatatypeKind, PayloadSource, PayloadType, SparseKind, SparseSource, VectorSource,
    default_collection_name, string_or_struct, validate_file_source_path,
};

/// Top-level document: `{ collection: { name }, requests: [ … ] }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    pub collection: SearchCollectionConfig,
    pub requests: Vec<SearchRequestConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCollectionConfig {
    #[serde(default = "default_collection_name")]
    pub name: String,
}

/// One search-request template. At benchmark time one template is picked at
/// random per batch; every query in the batch reuses it (with fresh random
/// vectors / filter values).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SearchRequestConfig {
    Dense {
        /// Named dense vector to query. Omit for the unnamed default vector.
        #[serde(default)]
        using: Option<String>,
        /// Vector dimension. Required for generated (`random`) queries; ignored
        /// when queries come from a dataset (the dataset defines the dimension).
        #[serde(default)]
        size: u64,
        #[serde(default)]
        datatype: DatatypeKind,
        #[serde(default, deserialize_with = "string_or_struct")]
        source: VectorSource,
        #[serde(default)]
        filters: Vec<FilterPayloadConfig>,
        /// Query with several sub-vectors of `size` each (a multivector). Random source only.
        #[serde(default)]
        multivector: Option<QueryMultivectorConfig>,
        /// Candidate stages. One prefetch feeds a rescore by this request's own vector;
        /// several feed a `fusion` query. Accepts a single entry or a list.
        #[serde(default, deserialize_with = "one_or_many")]
        prefetch: Vec<PrefetchConfig>,
        /// Combine several prefetches instead of rescoring them: `rrf` or `dbsf`.
        /// The request then sends no vector of its own.
        #[serde(default)]
        fusion: Option<FusionKind>,
        /// Reciprocal rank fusion's `k`. Needs `fusion: rrf`.
        #[serde(default)]
        rrf_k: Option<u32>,
        /// Per-prefetch weights, in the order the prefetches are listed. Needs `fusion: rrf`.
        #[serde(default)]
        weights: Option<Vec<f32>>,
    },
    Sparse {
        using: String,
        #[serde(default, deserialize_with = "string_or_struct")]
        source: SparseSource,
        #[serde(default)]
        filters: Vec<FilterPayloadConfig>,
        /// IDF corpus (Qdrant 1.19+): restricts the population sparse-vector IDF
        /// statistics are computed over to the points matching these conditions.
        /// Empty ⇒ collection-wide (global) statistics. Only meaningful for
        /// sparse vectors created with the IDF modifier.
        #[serde(default)]
        idf_corpus: Vec<FilterPayloadConfig>,
    },
}

/// Shape of a multivector query.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryMultivectorConfig {
    /// Sub-vectors per query.
    pub count: usize,
}

/// The first stage of a two-stage query: a random dense query on its own named vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefetchConfig {
    /// Named vector this stage searches.
    pub using: String,
    /// Dense only: its dimension.
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub datatype: DatatypeKind,
    /// Candidates this stage hands on.
    pub limit: u64,
    /// What the stage queries. Dense unless said otherwise.
    #[serde(default)]
    pub kind: PrefetchKind,
    /// Sparse only: how its query vectors are generated.
    #[serde(default)]
    pub source: Option<SparseSource>,
}

/// Which kind of vector a prefetch stage searches.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PrefetchKind {
    #[default]
    Dense,
    Sparse,
}

/// How several prefetches are combined into one ranking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FusionKind {
    /// Reciprocal Rank Fusion.
    Rrf,
    /// Distribution-Based Score Fusion.
    Dbsf,
}

/// A prefetch written as a single mapping, or several as a list.
fn one_or_many<'de, D>(deserializer: D) -> std::result::Result<Vec<PrefetchConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(Box<PrefetchConfig>),
        Many(Vec<PrefetchConfig>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(one) => vec![*one],
        OneOrMany::Many(many) => many,
    })
}

/// Payload field used to build a filter condition for a search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterPayloadConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: PayloadType,
    #[serde(default, deserialize_with = "string_or_struct")]
    pub source: PayloadSource,
    /// Keyword filters: match any of N random values instead of one.
    pub match_any: Option<usize>,
    /// Keyword filters: match a prefix of this many characters instead of a
    /// whole value. Requires the field's keyword index to be created with
    /// `prefix: true`. Takes precedence over `match_any`.
    pub match_prefix: Option<usize>,
}

impl FilterPayloadConfig {
    /// Validate one filter condition. `context` names where it came from, e.g.
    /// `"requests[0]"` or `"requests[0].idf_corpus"`.
    pub fn validate(&self, context: &str) -> Result<()> {
        if self.match_prefix.is_some() && self.kind != PayloadType::Keyword {
            bail!("{context}: `match_prefix` only applies to `type: keyword` filters");
        }
        if self.match_prefix == Some(0) {
            bail!("{context}: `match_prefix` must be > 0");
        }
        Ok(())
    }
}

/// Parse and validate a YAML search config.
pub fn parse(text: &str, origin: &str) -> Result<SearchConfig> {
    let config: SearchConfig = serde_yaml::from_str(text)
        .with_context(|| format!("failed to parse search config {origin}"))?;
    config.validate()?;
    Ok(config)
}

impl SearchConfig {
    /// Whether any request runs its query in stages: prefetch then rescore, or fusion.
    pub fn has_prefetch(&self) -> bool {
        self.requests.iter().any(|r| !r.prefetches().is_empty())
    }

    /// The smallest prefetch `limit` across requests, if any request prefetches.
    pub fn min_prefetch_limit(&self) -> Option<u64> {
        self.requests
            .iter()
            .flat_map(|r| r.prefetches().iter().map(|p| p.limit))
            .min()
    }

    pub fn validate(&self) -> Result<()> {
        if self.requests.is_empty() {
            bail!("search config must define at least one request");
        }

        for (i, req) in self.requests.iter().enumerate() {
            req.validate(i)?;
        }

        Ok(())
    }
}

impl SearchRequestConfig {
    /// Payload conditions applied to the query itself.
    /// The request's prefetch stages; empty when it queries in one step.
    pub fn prefetches(&self) -> &[PrefetchConfig] {
        match self {
            SearchRequestConfig::Dense { prefetch, .. } => prefetch,
            SearchRequestConfig::Sparse { .. } => &[],
        }
    }

    pub fn filters(&self) -> &[FilterPayloadConfig] {
        match self {
            SearchRequestConfig::Dense { filters, .. }
            | SearchRequestConfig::Sparse { filters, .. } => filters,
        }
    }

    fn validate(&self, index: usize) -> Result<()> {
        for filter in self.filters() {
            filter.validate(&format!("requests[{index}]"))?;
        }
        if let SearchRequestConfig::Sparse { idf_corpus, .. } = self {
            for filter in idf_corpus {
                filter.validate(&format!("requests[{index}].idf_corpus"))?;
            }
        }

        if let SearchRequestConfig::Dense {
            source,
            multivector,
            prefetch,
            fusion,
            rrf_k,
            weights,
            ..
        } = self
        {
            if let Some(multivector) = multivector {
                if multivector.count == 0 {
                    bail!("requests[{index}]: `multivector.count` must be > 0");
                }
                if !matches!(source, VectorSource::Random) {
                    bail!("requests[{index}]: `multivector` needs `source: random`");
                }
            }
            for (slot, stage) in prefetch.iter().enumerate() {
                let at = format!("requests[{index}].prefetch[{slot}]");
                if stage.using.is_empty() {
                    bail!("{at}: `using` must not be empty");
                }
                if stage.limit == 0 {
                    bail!("{at}: `limit` must be > 0");
                }
                match stage.kind {
                    PrefetchKind::Dense => {
                        // A misplaced `source` is the likelier mistake, so say that first.
                        if stage.source.is_some() {
                            bail!("{at}: `source` describes a sparse query; add `kind: sparse`");
                        }
                        if stage.size == 0 {
                            bail!("{at}: dense `size` must be > 0");
                        }
                    }
                    PrefetchKind::Sparse => {
                        let sparse = stage
                            .source
                            .as_ref()
                            .with_context(|| format!("{at}: a sparse stage needs `source`"))?;
                        if sparse.kind != SparseKind::Random {
                            bail!(
                                "{at}: sparse prefetch queries must be generated: `type: random`"
                            );
                        }
                        if sparse.length == 0 || sparse.length > sparse.vocab_size {
                            bail!("{at}: sparse `length` must be > 0 and <= `vocab_size`");
                        }
                    }
                }
            }
            if !prefetch.is_empty() && matches!(source, VectorSource::File { .. }) {
                bail!(
                    "requests[{index}]: `prefetch` needs `source: random` or a dataset \
                     source whose query set carries prefetch vectors"
                );
            }
            match fusion {
                None => {
                    if prefetch.len() > 1 {
                        bail!(
                            "requests[{index}]: {} prefetches need `fusion` to combine them; \
                             a rescore takes one",
                            prefetch.len()
                        );
                    }
                    if rrf_k.is_some() || weights.is_some() {
                        bail!("requests[{index}]: `rrf_k` and `weights` need `fusion: rrf`");
                    }
                }
                Some(kind) => {
                    if prefetch.len() < 2 {
                        bail!(
                            "requests[{index}]: `fusion` combines prefetches, so it needs at \
                             least 2; got {}",
                            prefetch.len()
                        );
                    }
                    if multivector.is_some() {
                        bail!(
                            "requests[{index}]: a `fusion` query sends no vector of its own, \
                             so `multivector` cannot apply"
                        );
                    }
                    if *kind == FusionKind::Dbsf && (rrf_k.is_some() || weights.is_some()) {
                        bail!("requests[{index}]: `rrf_k` and `weights` need `fusion: rrf`");
                    }
                    if let Some(weights) = weights
                        && weights.len() != prefetch.len()
                    {
                        bail!(
                            "requests[{index}]: {} `weights` for {} prefetches; one each",
                            weights.len(),
                            prefetch.len()
                        );
                    }
                }
            }
        }

        match self {
            SearchRequestConfig::Dense { size, source, .. } => {
                match source {
                    VectorSource::File { path, .. } => validate_file_source_path(path)
                        .with_context(|| format!("requests[{index}]"))?,
                    // A dataset query source supplies its own query vectors, so
                    // `size` is not required to match anything here.
                    VectorSource::Dataset { dataset } => dataset.validate_inline()?,
                    _ => {
                        if *size == 0 {
                            bail!("requests[{index}]: dense `size` must be > 0");
                        }
                    }
                }
            }
            SearchRequestConfig::Sparse { using, source, .. } => {
                if using.is_empty() {
                    bail!("requests[{index}]: sparse `using` must not be empty");
                }
                if source.kind == SparseKind::Dataset {
                    let dataset = source
                        .dataset
                        .as_ref()
                        .context("sparse dataset query source is missing dataset fields")?;
                    dataset.validate_inline()?;
                } else {
                    if source.length == 0 {
                        bail!("requests[{index}]: sparse `length` must be > 0");
                    }
                    if source.length > source.vocab_size {
                        bail!(
                            "requests[{index}]: sparse length ({}) must be <= vocab_size ({})",
                            source.length,
                            source.vocab_size
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DistributionKind;

    const RESCORE_YAML: &str = r#"
collection:
  name: bench
requests:
  - kind: dense
    using: colbert
    size: 128
    multivector: { count: 16 }
    prefetch: { using: dense, size: 128, limit: 500 }
"#;

    const FUSION_YAML: &str = "collection:\n  name: x\nrequests:\n  - kind: dense\n    size: 128\n    fusion: rrf\n    prefetch:\n      - using: dense\n        size: 128\n        limit: 200\n      - using: bm25\n        kind: sparse\n        limit: 200\n        source: { vocab_size: 1000, length: 10 }\n";

    #[test]
    fn parses_hybrid_fusion_request() {
        let cfg: SearchConfig = serde_yaml::from_str(FUSION_YAML).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.has_prefetch());
        assert_eq!(cfg.min_prefetch_limit(), Some(200));
        match &cfg.requests[0] {
            SearchRequestConfig::Dense {
                prefetch, fusion, ..
            } => {
                assert_eq!(*fusion, Some(FusionKind::Rrf));
                assert_eq!(prefetch.len(), 2);
                assert_eq!(prefetch[0].kind, PrefetchKind::Dense);
                assert_eq!(prefetch[1].kind, PrefetchKind::Sparse);
                assert_eq!(prefetch[1].source.as_ref().unwrap().length, 10);
            }
            other => panic!("unexpected request {other:?}"),
        }
    }

    #[test]
    fn parses_rrf_parameters() {
        let yaml = FUSION_YAML.replace(
            "fusion: rrf",
            "fusion: rrf\n    rrf_k: 60\n    weights: [2.0, 1.0]",
        );
        let cfg: SearchConfig = serde_yaml::from_str(&yaml).unwrap();
        cfg.validate().unwrap();
        match &cfg.requests[0] {
            SearchRequestConfig::Dense { rrf_k, weights, .. } => {
                assert_eq!(*rrf_k, Some(60));
                assert_eq!(weights.as_deref(), Some(&[2.0f32, 1.0][..]));
            }
            other => panic!("unexpected request {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_fusion() {
        for (from, to, message) in [
            // Several stages with nothing to combine them.
            ("    fusion: rrf\n", "", "need `fusion` to combine them"),
            // Fusion ranks stages against each other, so one is not a fusion.
            (
                "      - using: bm25\n        kind: sparse\n        limit: 200\n        source: { vocab_size: 1000, length: 10 }\n",
                "",
                "needs at least 2",
            ),
            // One weight per stage, or the ranking silently means something else.
            (
                "fusion: rrf",
                "fusion: rrf\n    weights: [1.0]",
                "1 `weights` for 2 prefetches",
            ),
            (
                "fusion: rrf",
                "fusion: dbsf\n    rrf_k: 60",
                "need `fusion: rrf`",
            ),
            (
                "        kind: sparse\n",
                "",
                "`source` describes a sparse query; add `kind: sparse`",
            ),
            (
                "        source: { vocab_size: 1000, length: 10 }\n",
                "",
                "a sparse stage needs `source`",
            ),
            (
                "    size: 128\n    fusion: rrf",
                "    size: 128\n    multivector: { count: 4 }\n    fusion: rrf",
                "sends no vector of its own",
            ),
        ] {
            let yaml = FUSION_YAML.replace(from, to);
            assert_ne!(yaml, FUSION_YAML, "replacement {from:?} did not apply");
            let err = match serde_yaml::from_str::<SearchConfig>(&yaml) {
                Ok(cfg) => match cfg.validate() {
                    Err(err) => err.to_string(),
                    Ok(()) => panic!("accepted {message:?}"),
                },
                Err(err) => err.to_string(),
            };
            assert!(err.contains(message), "expected {message:?}, got {err}");
        }
    }

    #[test]
    fn parses_multivector_prefetch_request() {
        let cfg: SearchConfig = serde_yaml::from_str(RESCORE_YAML).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.has_prefetch());
        match &cfg.requests[0] {
            SearchRequestConfig::Dense {
                multivector: Some(m),
                prefetch,
                ..
            } => {
                assert_eq!(m.count, 16);
                let [p] = &prefetch[..] else {
                    panic!("expected one prefetch, got {}", prefetch.len())
                };
                assert_eq!((p.using.as_str(), p.size, p.limit), ("dense", 128, 500));
            }
            other => panic!("unexpected request {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_multivector_and_prefetch() {
        for (from, to, message) in [
            ("count: 16", "count: 0", "`multivector.count` must be > 0"),
            ("limit: 500", "limit: 0", "`limit` must be > 0"),
            (
                "{ using: dense,",
                "{ using: \"\",",
                "`using` must not be empty",
            ),
            (
                "size: 128, limit",
                "size: 0, limit",
                "dense `size` must be > 0",
            ),
            (
                "multivector: { count: 16 }",
                "multivector: { count: 16 }\n    source: { type: file, path: q.fbin }",
                "`multivector` needs `source: random`",
            ),
        ] {
            let yaml = RESCORE_YAML.replace(from, to);
            let cfg: SearchConfig = serde_yaml::from_str(&yaml).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains(message), "{from} -> {to}: {err}");
        }
    }

    #[test]
    fn parses_minimal_search_config() {
        let yaml = r#"
collection:
  name: test
requests:
  - kind: dense
    size: 128
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.collection.name, "test");
        assert_eq!(cfg.requests.len(), 1);
    }

    #[test]
    fn parses_dense_sparse_and_filters() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    using: image
    size: 512
    source: random
  - kind: sparse
    using: bm25
    source: { type: random, vocab_size: 1000, length: 100, distribution: zipf }
  - kind: dense
    using: image
    size: 512
    filters:
      - name: color
        type: keyword
        source: { cardinality: 100 }
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert!(matches!(cfg.requests[0], SearchRequestConfig::Dense { .. }));
        match &cfg.requests[1] {
            SearchRequestConfig::Sparse { source, .. } => {
                assert_eq!(source.distribution, DistributionKind::Zipf);
            }
            _ => panic!("expected sparse request"),
        }
        match &cfg.requests[2] {
            SearchRequestConfig::Dense { filters, .. } => assert_eq!(filters.len(), 1),
            _ => panic!("expected dense request with filters"),
        }
    }

    #[test]
    fn parses_dataset_query_sources() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    source:
      type: dataset
      name: glove-25-angular
      format: h5
      path: glove-25-angular/glove-25-angular.hdf5
      link: http://ann-benchmarks.com/glove-25-angular.hdf5
  - kind: sparse
    using: bm25
    source:
      type: dataset
      dataset:
        name: my-sparse
        format: sparse
        path: my-sparse/my-sparse
        link: https://example.com/my-sparse.tgz
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert!(matches!(
            cfg.requests[0],
            SearchRequestConfig::Dense {
                source: VectorSource::Dataset { .. },
                ..
            }
        ));
    }

    #[test]
    fn rejects_dense_dataset_without_format() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    source:
      type: dataset
      name: glove-25-angular
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }

    fn search_config_with_vector_path(path: &str) -> SearchConfig {
        let yaml = format!(
            r#"
collection:
  name: bench
requests:
  - kind: dense
    size: 128
    source:
      type: file
      path: {path}
"#
        );
        serde_yaml::from_str(&yaml).unwrap()
    }

    #[test]
    fn accepts_http_query_vector_path() {
        search_config_with_vector_path("https://example.com/queries.fbin")
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_s3_query_vector_path() {
        let err = search_config_with_vector_path("s3://bucket/queries.fbin")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("requests[0]"), "{err}");
    }

    #[test]
    fn parses_match_prefix_and_idf_corpus() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: sparse
    using: bm25
    filters:
      - name: color
        type: keyword
        source: { cardinality: 100 }
        match_prefix: 9
    idf_corpus:
      - name: tenant
        type: keyword
        source: { cardinality: 10 }
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        match &cfg.requests[0] {
            SearchRequestConfig::Sparse {
                filters,
                idf_corpus,
                ..
            } => {
                assert_eq!(filters[0].match_prefix, Some(9));
                assert_eq!(idf_corpus.len(), 1);
                assert_eq!(idf_corpus[0].name, "tenant");
            }
            _ => panic!("expected sparse request"),
        }
    }

    #[test]
    fn rejects_match_prefix_on_non_keyword_field() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    size: 4
    filters:
      - name: age
        type: integer
        match_prefix: 3
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("match_prefix"), "{err}");
    }

    #[test]
    fn rejects_zero_length_match_prefix() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    size: 4
    filters:
      - name: color
        type: keyword
        match_prefix: 0
"#;
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_idf_corpus_on_dense_requests() {
        let yaml = r#"
collection:
  name: bench
requests:
  - kind: dense
    size: 4
    idf_corpus: []
"#;
        assert!(serde_yaml::from_str::<SearchConfig>(yaml).is_err());
    }

    /// Same contract as the upload example: the shipped search config must stay
    /// loadable.
    #[test]
    fn parses_search_config_example() {
        let cfg = crate::config::examples::lookup("search-config")
            .map(|e| super::parse(e.yaml, e.name).unwrap())
            .unwrap();
        assert!(
            cfg.requests.iter().any(|r| matches!(
                r,
                SearchRequestConfig::Sparse { idf_corpus, .. } if !idf_corpus.is_empty()
            )),
            "example lost its idf_corpus request"
        );
        assert!(
            cfg.requests
                .iter()
                .flat_map(SearchRequestConfig::filters)
                .any(|f| f.match_prefix.is_some()),
            "example lost its match_prefix filter"
        );
    }

    #[test]
    fn rejects_empty_requests() {
        let yaml = "collection:\n  name: x\nrequests: []\n";
        let cfg: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_unknown_field() {
        let yaml = r#"
collection:
  name: x
requests:
  - kind: dense
    size: 128
    bogus: 1
"#;
        assert!(serde_yaml::from_str::<SearchConfig>(yaml).is_err());
    }
}
