//! Search query generation for the YAML-config path ([`ConfigSearchGenerator`]).

use std::collections::HashSet;
use std::path::Path;

use qdrant_client::qdrant::{
    Condition, Filter, GeoPoint, GeoRadius, Range, RepeatedStrings, SparseIndices,
    r#match::MatchValue,
};
use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::seq::IndexedRandom;

use crate::common::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_dense_vector, random_keyword, random_sparse_vector,
    random_text,
};
use crate::config::{
    DatatypeKind, DistributionKind, FileStrategy, PayloadSourceKind, PayloadType, VectorSource,
};
use crate::fbin_reader::FBinReader;
use crate::search_config::{FilterPayloadConfig, SearchConfig, SearchRequestConfig};

const GEO_CENTER_LAT: f64 = 52.52437;
const GEO_CENTER_LON: f64 = 13.41053;
const GEO_SPREAD_DEG: f64 = 1.0;
const GEO_RADIUS_METERS_MIN: f64 = 1000.0;
const GEO_RADIUS_METERS_MAX: f64 = 50000.0;

/// One query vector plus optional filter, ready to be turned into a gRPC request.
#[derive(Debug, Clone)]
pub struct GeneratedQuery {
    pub dense: Option<(Vec<f32>, Option<String>)>,
    pub sparse: Option<(Vec<f32>, SparseIndices, String)>,
    pub filter: Option<Filter>,
}

struct RequestState {
    dense_reader: Option<FBinReader>,
    sparse_zipf: Option<rand_distr::Zipf<f64>>,
    text_zipf: Vec<Option<rand_distr::Zipf<f64>>>,
    geo_clusters: Vec<Option<Vec<(f64, f64)>>>,
}

/// Generates search queries from a parsed YAML [`SearchConfig`].
pub struct ConfigSearchGenerator {
    config: SearchConfig,
    per_request: Vec<RequestState>,
}

impl ConfigSearchGenerator {
    pub fn new(config: &SearchConfig) -> anyhow::Result<Self> {
        let mut rng = rand::rng();
        let mut per_request = Vec::with_capacity(config.requests.len());

        for req in &config.requests {
            let (dense_reader, sparse_zipf, filters) = match req {
                SearchRequestConfig::Dense {
                    source, filters, ..
                } => (
                    match source {
                        VectorSource::File { path, .. } => Some(FBinReader::new(Path::new(path))),
                        VectorSource::Random | VectorSource::Dataset { .. } => None,
                    },
                    None,
                    filters,
                ),
                SearchRequestConfig::Sparse {
                    source, filters, ..
                } => (
                    None,
                    (source.distribution == DistributionKind::Zipf)
                        .then(|| create_zipf(source.vocab_size)),
                    filters,
                ),
            };

            let text_zipf = filters.iter().map(Self::maybe_text_zipf).collect();
            let geo_clusters = filters
                .iter()
                .map(|f| Self::maybe_geo_clusters(f, &mut rng))
                .collect();

            per_request.push(RequestState {
                dense_reader,
                sparse_zipf,
                text_zipf,
                geo_clusters,
            });
        }

        Ok(ConfigSearchGenerator {
            config: config.clone(),
            per_request,
        })
    }

    fn maybe_text_zipf(filter: &FilterPayloadConfig) -> Option<rand_distr::Zipf<f64>> {
        (filter.kind == PayloadType::Text && filter.source.distribution == DistributionKind::Zipf)
            .then(|| create_zipf(filter.source.vocab_size.unwrap_or(DEFAULT_VOCAB_SIZE)))
    }

    fn maybe_geo_clusters(
        filter: &FilterPayloadConfig,
        rng: &mut impl Rng,
    ) -> Option<Vec<(f64, f64)>> {
        (filter.kind == PayloadType::Geo && filter.source.kind == PayloadSourceKind::RandomClusters)
            .then(|| {
                let count = filter.source.clusters.unwrap_or(10);
                (0..count)
                    .map(|_| {
                        (
                            GEO_CENTER_LAT + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                            GEO_CENTER_LON + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                        )
                    })
                    .collect()
            })
    }

    /// Pick a random request template and materialize one query from it.
    #[allow(dead_code)]
    pub fn make_query(&self, req_id: usize, rng: &mut impl Rng) -> GeneratedQuery {
        let template_idx = self.random_template_idx(rng);
        self.make_query_for(template_idx, req_id, rng)
    }

    pub fn random_template_idx(&self, rng: &mut impl Rng) -> usize {
        rng.random_range(0..self.config.requests.len())
    }

    /// Materialize a query from a specific request template.
    pub fn make_query_for(
        &self,
        template_idx: usize,
        req_id: usize,
        rng: &mut impl Rng,
    ) -> GeneratedQuery {
        let template = &self.config.requests[template_idx];
        let state = &self.per_request[template_idx];

        match template {
            SearchRequestConfig::Dense {
                using,
                size,
                datatype,
                source,
                filters,
            } => {
                let vector = Self::gen_dense_vector(
                    rng,
                    *size as usize,
                    *datatype,
                    source,
                    state.dense_reader.as_ref(),
                    req_id,
                );
                GeneratedQuery {
                    dense: Some((vector, using.clone())),
                    sparse: None,
                    filter: self.build_filter(rng, filters, state),
                }
            }
            SearchRequestConfig::Sparse {
                using,
                source,
                filters,
            } => {
                let (values, indices) =
                    Self::gen_sparse_vector(rng, source, state.sparse_zipf.as_ref());
                GeneratedQuery {
                    dense: None,
                    sparse: Some((values, indices, using.clone())),
                    filter: self.build_filter(rng, filters, state),
                }
            }
        }
    }

    fn gen_dense_vector(
        rng: &mut impl Rng,
        size: usize,
        datatype: DatatypeKind,
        source: &VectorSource,
        reader: Option<&FBinReader>,
        req_id: usize,
    ) -> Vec<f32> {
        match (source, reader) {
            (VectorSource::File { strategy, .. }, Some(reader)) => {
                let n = reader.num_vectors.max(1) as usize;
                let file_idx = match strategy {
                    FileStrategy::FromStart => req_id % n,
                    FileStrategy::RandomSample => rng.random_range(0..n),
                };
                reader.read_vector(file_idx).to_vec()
            }
            _ => {
                let is_uint = datatype == DatatypeKind::Uint8;
                random_dense_vector(rng, size, is_uint)
            }
        }
    }

    fn gen_sparse_vector(
        rng: &mut impl Rng,
        source: &crate::config::SparseSource,
        zipf: Option<&rand_distr::Zipf<f64>>,
    ) -> (Vec<f32>, SparseIndices) {
        match zipf {
            Some(zipf) => {
                let target = source.length;
                let mut seen = HashSet::new();
                let mut pairs = Vec::with_capacity(target);
                let mut attempts = 0;
                while pairs.len() < target && attempts < target * 8 {
                    attempts += 1;
                    let dim = (zipf.sample(rng) as u32).max(1);
                    if seen.insert(dim) {
                        pairs.push((dim, rng.random_range(0.0..10.0) as f32));
                    }
                }
                let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                (values, SparseIndices { data: indices })
            }
            None => {
                let pairs = random_sparse_vector(rng, source.vocab_size, source.length);
                let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                (values, SparseIndices { data: indices })
            }
        }
    }

    fn build_filter(
        &self,
        rng: &mut impl Rng,
        filters: &[FilterPayloadConfig],
        state: &RequestState,
    ) -> Option<Filter> {
        if filters.is_empty() {
            return None;
        }

        let mut filter = Filter {
            should: vec![],
            must: vec![],
            must_not: vec![],
            min_should: None,
        };

        for (i, fp) in filters.iter().enumerate() {
            let src = &fp.source;
            match fp.kind {
                PayloadType::Keyword => {
                    let card = src.cardinality.unwrap_or(100);
                    let mult = src.length_multiplier.unwrap_or(1);
                    let condition = if let Some(match_any) = fp.match_any {
                        MatchValue::Keywords(RepeatedStrings {
                            strings: (0..match_any)
                                .map(|_| random_keyword(rng, card, mult))
                                .collect(),
                        })
                    } else {
                        MatchValue::Keyword(random_keyword(rng, card, mult))
                    };
                    filter.must.push(Condition::matches(&fp.name, condition));
                }
                PayloadType::Integer => {
                    let min = src.min.unwrap_or(0.0) as i64;
                    let max = src.max.unwrap_or(100.0) as i64;
                    let max = max.max(min + 1);
                    let rand_int = rng.random_range(min..max);
                    filter.must.push(Condition::range(
                        &fp.name,
                        Range {
                            gt: None,
                            gte: Some(rand_int as f64),
                            lt: None,
                            lte: None,
                        },
                    ));
                }
                PayloadType::Float => {
                    filter.must.push(Condition::range(
                        &fp.name,
                        Range {
                            gt: None,
                            gte: Some(0.0),
                            lt: None,
                            lte: None,
                        },
                    ));
                }
                PayloadType::Bool => {
                    filter.must.push(Condition::matches(
                        &fp.name,
                        rng.random_bool(src.true_ratio.unwrap_or(0.5)),
                    ));
                }
                PayloadType::Uuid => {
                    filter.must.push(Condition::matches(
                        &fp.name,
                        uuid::Uuid::new_v4().to_string(),
                    ));
                }
                PayloadType::Geo => {
                    let (lat, lon) = match state.geo_clusters.get(i).and_then(|c| c.as_ref()) {
                        Some(centers) => {
                            let &(clat, clon) = centers.choose(rng).unwrap();
                            (
                                clat + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                                clon + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                            )
                        }
                        None => (
                            GEO_CENTER_LAT + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                            GEO_CENTER_LON + rng.random_range(-GEO_SPREAD_DEG..GEO_SPREAD_DEG),
                        ),
                    };
                    let radius = rng.random_range(GEO_RADIUS_METERS_MIN..GEO_RADIUS_METERS_MAX);
                    filter.must.push(Condition::geo_radius(
                        &fp.name,
                        GeoRadius {
                            center: Some(GeoPoint { lat, lon }),
                            radius: radius as f32,
                        },
                    ));
                }
                PayloadType::Text => {
                    let min_len = src.min_length.unwrap_or(2);
                    let max_len = src.max_length.unwrap_or(min_len).max(min_len);
                    let len = if max_len > min_len {
                        rng.random_range(min_len..=max_len)
                    } else {
                        min_len
                    };
                    let text = match state.text_zipf.get(i).and_then(|z| z.as_ref()) {
                        Some(zipf) => random_text(rng, len, zipf),
                        None => (0..len)
                            .map(|_| {
                                format!(
                                    "word_{}",
                                    rng.random_range(
                                        0..src.vocab_size.unwrap_or(DEFAULT_VOCAB_SIZE)
                                    )
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                    };
                    filter.must.push(Condition::matches_text(&fp.name, text));
                }
                PayloadType::Datetime => {}
            }
        }

        if filter.must.is_empty() {
            None
        } else {
            Some(filter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gen(yaml: &str) -> ConfigSearchGenerator {
        let config: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        ConfigSearchGenerator::new(&config).unwrap()
    }

    #[test]
    fn generates_dense_and_sparse_queries() {
        let generator = build_gen(
            "collection:\n  name: x\nrequests:\n  - kind: dense\n    size: 8\n  - kind: sparse\n    using: bm25\n    source: { vocab_size: 32, length: 6 }\n",
        );
        let mut rng = rand::rng();

        for _ in 0..20 {
            let q = generator.make_query(0, &mut rng);
            assert!(q.dense.is_some() ^ q.sparse.is_some());
        }
    }

    #[test]
    fn generates_keyword_filter() {
        let generator = build_gen(
            "collection:\n  name: x\nrequests:\n  - kind: dense\n    size: 4\n    filters:\n      - name: color\n        type: keyword\n        source: { cardinality: 5 }\n",
        );
        let mut rng = rand::rng();
        let q = generator.make_query(0, &mut rng);
        assert!(q.filter.is_some());
        assert!(!q.filter.as_ref().unwrap().must.is_empty());
    }
}
