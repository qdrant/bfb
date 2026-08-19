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
