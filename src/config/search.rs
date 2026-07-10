//! YAML search-request configuration for `bfb search --file config.yaml`.
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
}

/// Read and validate a YAML search config file.
pub fn load(path: &str) -> Result<SearchConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read search config file {path}"))?;
    let config: SearchConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse search config file {path}"))?;
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
    fn validate(&self, index: usize) -> Result<()> {
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
