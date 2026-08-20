//! Search query generation for the YAML-config path ([`ConfigSearchGenerator`]).

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Context;
use qdrant_client::qdrant::{
    Condition, Filter, GeoPoint, GeoRadius, Range, RepeatedStrings, SparseIndices,
    r#match::MatchValue,
};
use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::seq::IndexedRandom;

use super::random::{
    DEFAULT_VOCAB_SIZE, create_zipf, random_dense_vector, random_keyword, random_sparse_vector,
    random_text,
};
use crate::config::search::{FilterPayloadConfig, SearchConfig, SearchRequestConfig};
use crate::config::{
    DatatypeKind, DistributionKind, FileStrategy, PayloadSourceKind, PayloadType, SparseKind,
    VectorSource,
};
use crate::dataset::{DatasetReader, default_datasets_dir, ensure_local_file};
use crate::fbin_reader::FBinReader;

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
    /// Sparse-vector IDF corpus: restricts which points the IDF statistics are
    /// computed over. `None` ⇒ collection-wide (global) statistics.
    pub idf_corpus: Option<Filter>,
    /// Ground-truth nearest-neighbor point ids for this query, present only
    /// when the request draws queries from a reference dataset. Used to measure
    /// search accuracy (recall) against the dataset's known answers.
    pub expected_ids: Option<Vec<u64>>,
}

/// A reference dataset's query set, held in memory, with a cursor that hands out
/// consecutive query indices so every query in the set is exercised.
///
/// The whole set is read once when the dataset is opened rather than a row at a
/// time during the benchmark. Resolving a row on demand meant reopening the
/// file and parsing the JSON twice per search — for a 2048-d query that is
/// ~45 KB of text each way, enough client-side work to outweigh the search being
/// timed. Parsed form is far smaller than the file: a 10k x 2048-d set is ~82 MB.
struct QueryDataset {
    vectors: QueryVectors,
    /// Ground-truth nearest-neighbor ids per query, used to score recall.
    ground_truth: Vec<Vec<u64>>,
    /// Filter each query was answered under, parallel to `vectors`. Its ground
    /// truth only holds for a search that applies it, so this is not optional
    /// decoration — dropping it scores the search against the answers to a
    /// different question.
    filters: Vec<Option<Filter>>,
    num_queries: usize,
    cursor: AtomicUsize,
}

/// Query vectors in whichever form the request that opened the dataset needs.
enum QueryVectors {
    Dense(Vec<Vec<f32>>),
    /// Kept pre-split as (values, indices) so no per-request unzip is needed.
    Sparse(Vec<(Vec<f32>, Vec<u32>)>),
}

/// A sparse query drawn from a dataset query set: the `(values, indices)`
/// vector, the ground truth it is scored against, and the filter that ground
/// truth assumes.
type SparseDatasetQuery = ((Vec<f32>, SparseIndices), Option<Vec<u64>>, Option<Filter>);

/// Which kind of query a request will draw from a dataset.
#[derive(Clone, Copy)]
enum QueryKind {
    Dense,
    Sparse,
}

impl QueryDataset {
    /// Next query index to use, wrapping around the query set.
    fn next_index(&self) -> usize {
        self.cursor.fetch_add(1, Ordering::Relaxed) % self.num_queries
    }
}

struct RequestState {
    dense_reader: Option<FBinReader>,
    /// Reference dataset used as the query source (dense or sparse).
    query_dataset: Option<QueryDataset>,
    sparse_zipf: Option<rand_distr::Zipf<f64>>,
    filters: FilterGenerator,
    /// Sparse requests only: conditions defining the IDF corpus.
    idf_corpus: FilterGenerator,
}

/// Builds a random payload filter from a list of [`FilterPayloadConfig`].
///
/// Shared by `bfb search --file` and `bfb scroll --file`: the filter half of a
/// request is the same in both, only the query half differs.
pub struct FilterGenerator {
    filters: Vec<FilterPayloadConfig>,
    /// Parallel to `filters`; `Some` only for zipf-distributed text fields.
    text_zipf: Vec<Option<rand_distr::Zipf<f64>>>,
    /// Parallel to `filters`; `Some` only for clustered geo fields.
    geo_clusters: Vec<Option<Vec<(f64, f64)>>>,
}

impl FilterGenerator {
    pub fn new(filters: &[FilterPayloadConfig], rng: &mut impl Rng) -> Self {
        FilterGenerator {
            filters: filters.to_vec(),
            text_zipf: filters.iter().map(Self::maybe_text_zipf).collect(),
            geo_clusters: filters
                .iter()
                .map(|f| Self::maybe_geo_clusters(f, rng))
                .collect(),
        }
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

    /// Materialize one random filter. `None` when no filters are configured.
    pub fn build(&self, rng: &mut impl Rng) -> Option<Filter> {
        if self.filters.is_empty() {
            return None;
        }

        let mut filter = Filter {
            should: vec![],
            must: vec![],
            must_not: vec![],
            min_should: None,
        };

        for (i, fp) in self.filters.iter().enumerate() {
            let src = &fp.source;
            match fp.kind {
                PayloadType::Keyword => {
                    let card = src.cardinality.unwrap_or(100);
                    let mult = src.length_multiplier.unwrap_or(1);
                    let condition = if let Some(len) = fp.match_prefix {
                        // Truncate a generated value: shorter prefixes match more
                        // keywords, which is the selectivity knob here.
                        let keyword = random_keyword(rng, card, mult);
                        let prefix: String = keyword.chars().take(len).collect();
                        MatchValue::Prefix(prefix)
                    } else if let Some(match_any) = fp.match_any {
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
                    let (lat, lon) = match self.geo_clusters.get(i).and_then(|c| c.as_ref()) {
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
                    let text = match self.text_zipf.get(i).and_then(|z| z.as_ref()) {
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

/// Generates search queries from a parsed YAML [`SearchConfig`].
pub struct ConfigSearchGenerator {
    config: SearchConfig,
    per_request: Vec<RequestState>,
}

impl ConfigSearchGenerator {
    pub fn new(config: &SearchConfig) -> anyhow::Result<Self> {
        Self::new_with_datasets_dir(config, &default_datasets_dir())
    }

    pub fn new_with_datasets_dir(
        config: &SearchConfig,
        datasets_dir: &Path,
    ) -> anyhow::Result<Self> {
        let mut rng = rand::rng();
        let mut per_request = Vec::with_capacity(config.requests.len());

        for req in &config.requests {
            let (dense_reader, query_dataset, sparse_zipf, filters, idf_corpus) = match req {
                SearchRequestConfig::Dense {
                    source, filters, ..
                } => {
                    let (dense_reader, query_dataset) = match source {
                        VectorSource::File { path, .. } => {
                            let local = ensure_local_file(datasets_dir, path)?;
                            (Some(FBinReader::new(&local)?), None)
                        }
                        VectorSource::Dataset { dataset } => (
                            None,
                            Some(Self::open_query_dataset(
                                dataset,
                                datasets_dir,
                                QueryKind::Dense,
                            )?),
                        ),
                        VectorSource::Random => (None, None),
                    };
                    (dense_reader, query_dataset, None, filters, [].as_slice())
                }
                SearchRequestConfig::Sparse {
                    source,
                    filters,
                    idf_corpus,
                    ..
                } => {
                    if source.kind == SparseKind::Dataset {
                        let dataset = source
                            .dataset
                            .as_ref()
                            .context("sparse dataset query source is missing dataset fields")?;
                        (
                            None,
                            Some(Self::open_query_dataset(
                                dataset,
                                datasets_dir,
                                QueryKind::Sparse,
                            )?),
                            None,
                            filters,
                            idf_corpus.as_slice(),
                        )
                    } else {
                        (
                            None,
                            None,
                            (source.distribution == DistributionKind::Zipf)
                                .then(|| create_zipf(source.vocab_size)),
                            filters,
                            idf_corpus.as_slice(),
                        )
                    }
                }
            };

            per_request.push(RequestState {
                dense_reader,
                query_dataset,
                sparse_zipf,
                filters: FilterGenerator::new(filters, &mut rng),
                idf_corpus: FilterGenerator::new(idf_corpus, &mut rng),
            });
        }

        Ok(ConfigSearchGenerator {
            config: config.clone(),
            per_request,
        })
    }

    /// Open a reference dataset as a query source and read its entire query set
    /// into memory, requiring it to ship one (ann-benchmarks `test`/`neighbors`,
    /// or `tests.jsonl` / `queries.csr`+`results.gt`).
    ///
    /// Reading it all here is what keeps file I/O and JSON parsing out of the
    /// timed request path, so a query set that is missing, truncated, or of the
    /// wrong kind fails at startup rather than part-way through a benchmark.
    /// Turn one query row's raw `conditions` into a filter, naming the row on
    /// failure so a bad line in a 10k-query set is findable.
    fn parse_conditions(
        conditions: &Option<serde_json::Value>,
        dataset: &str,
        idx: usize,
    ) -> anyhow::Result<Option<Filter>> {
        crate::dataset::parse_query_conditions(conditions.as_ref())
            .with_context(|| format!("dataset {dataset:?}, query {idx}"))
    }

    fn open_query_dataset(
        dataset: &crate::dataset::DatasetConfig,
        datasets_dir: &Path,
        kind: QueryKind,
    ) -> anyhow::Result<QueryDataset> {
        let reader = DatasetReader::open(datasets_dir, dataset)?;
        let num_queries = reader.num_queries();
        if num_queries == 0 {
            anyhow::bail!(
                "dataset {:?} has no query set; cannot use it as a search query source",
                dataset.name
            );
        }

        // Conditions are parsed once here, so a malformed query set fails at
        // startup and the timed path only ever clones a ready-made filter.
        let mut filters: Vec<Option<Filter>> = Vec::with_capacity(num_queries);
        let (vectors, ground_truth) = match kind {
            QueryKind::Dense => {
                let rows = reader.read_dense_query_set().with_context(|| {
                    format!(
                        "failed to read dense query set of dataset {:?}",
                        dataset.name
                    )
                })?;
                let mut vectors = Vec::with_capacity(rows.len());
                let mut ground_truth = Vec::with_capacity(rows.len());
                for (idx, row) in rows.into_iter().enumerate() {
                    vectors.push(row.vector);
                    ground_truth.push(row.ground_truth);
                    filters.push(Self::parse_conditions(&row.conditions, &dataset.name, idx)?);
                }
                (QueryVectors::Dense(vectors), ground_truth)
            }
            QueryKind::Sparse => {
                let rows = reader.read_sparse_query_set().with_context(|| {
                    format!(
                        "failed to read sparse query set of dataset {:?}",
                        dataset.name
                    )
                })?;
                let mut vectors = Vec::with_capacity(rows.len());
                let mut ground_truth = Vec::with_capacity(rows.len());
                for (idx, row) in rows.into_iter().enumerate() {
                    // Pre-split into (values, indices) so no per-request unzip is needed.
                    let (indices, values): (Vec<u32>, Vec<f32>) = row.vector.into_iter().unzip();
                    vectors.push((values, indices));
                    ground_truth.push(row.ground_truth);
                    filters.push(Self::parse_conditions(&row.conditions, &dataset.name, idx)?);
                }
                (QueryVectors::Sparse(vectors), ground_truth)
            }
        };

        let filtered = filters.iter().filter(|f| f.is_some()).count();
        if filtered > 0 {
            println!(
                "Dataset {:?}: {filtered} of {num_queries} queries carry filter \
                 conditions; applying them (their ground truth assumes it).",
                dataset.name
            );
        }

        Ok(QueryDataset {
            vectors,
            ground_truth,
            filters,
            num_queries,
            cursor: AtomicUsize::new(0),
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
                filters: _,
            } => {
                let (vector, expected_ids, dataset_filter) =
                    if let Some(query_dataset) = &state.query_dataset {
                        Self::read_dense_query(query_dataset)
                    } else {
                        let vector = Self::gen_dense_vector(
                            rng,
                            *size as usize,
                            *datatype,
                            source,
                            state.dense_reader.as_ref(),
                            req_id,
                        );
                        (vector, None, None)
                    };
                GeneratedQuery {
                    dense: Some((vector, using.clone())),
                    sparse: None,
                    filter: dataset_filter.or_else(|| state.filters.build(rng)),
                    idf_corpus: None,
                    expected_ids,
                }
            }
            SearchRequestConfig::Sparse {
                using,
                source,
                filters: _,
                idf_corpus: _,
            } => {
                let ((values, indices), expected_ids, dataset_filter) =
                    if let Some(query_dataset) = &state.query_dataset {
                        Self::read_sparse_query(query_dataset)
                    } else {
                        (
                            Self::gen_sparse_vector(rng, source, state.sparse_zipf.as_ref()),
                            None,
                            None,
                        )
                    };
                GeneratedQuery {
                    dense: None,
                    sparse: Some((values, indices, using.clone())),
                    filter: dataset_filter.or_else(|| state.filters.build(rng)),
                    idf_corpus: state.idf_corpus.build(rng),
                    expected_ids,
                }
            }
        }
    }

    /// Take the next dense query vector and its ground-truth ids from a dataset.
    ///
    /// The kind mismatch cannot happen: the request template that opened the
    /// dataset is the same one reading from it here.
    fn read_dense_query(
        query_dataset: &QueryDataset,
    ) -> (Vec<f32>, Option<Vec<u64>>, Option<Filter>) {
        let idx = query_dataset.next_index();
        let QueryVectors::Dense(vectors) = &query_dataset.vectors else {
            panic!("dense request drew from a query set opened as sparse");
        };
        (
            vectors[idx].clone(),
            Some(query_dataset.ground_truth[idx].clone()),
            query_dataset.filters[idx].clone(),
        )
    }

    /// Take the next sparse query vector and its ground-truth ids from a dataset.
    fn read_sparse_query(query_dataset: &QueryDataset) -> SparseDatasetQuery {
        let idx = query_dataset.next_index();
        let QueryVectors::Sparse(vectors) = &query_dataset.vectors else {
            panic!("sparse request drew from a query set opened as dense");
        };
        let (values, indices) = &vectors[idx];
        (
            (
                values.clone(),
                SparseIndices {
                    data: indices.clone(),
                },
            ),
            Some(query_dataset.ground_truth[idx].clone()),
            query_dataset.filters[idx].clone(),
        )
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
}

#[cfg(test)]
mod tests {
    use qdrant_client::qdrant::condition::ConditionOneOf;

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
    fn dataset_query_source_yields_ground_truth() {
        use hdf5_pure_rust::WritableFile;

        let datasets_dir = tempfile::tempdir().unwrap();
        let h5_path = datasets_dir.path().join("glove/glove.hdf5");
        std::fs::create_dir_all(h5_path.parent().unwrap()).unwrap();
        {
            let mut wf = WritableFile::create(&h5_path).unwrap();
            let train: Vec<f32> = (0..12).map(|x| x as f32).collect();
            wf.new_dataset_builder("train")
                .shape(&[3, 4])
                .write(&train)
                .unwrap();
            let test: Vec<f32> = (0..8).map(|x| x as f32).collect();
            wf.new_dataset_builder("test")
                .shape(&[2, 4])
                .write(&test)
                .unwrap();
            let neighbors: Vec<i32> = vec![0, 2, 1, 2];
            wf.new_dataset_builder("neighbors")
                .shape(&[2, 2])
                .write(&neighbors)
                .unwrap();
            wf.close().unwrap();
        }

        let yaml = "collection:\n  name: x\nrequests:\n  - kind: dense\n    source:\n      type: dataset\n      name: glove\n      format: h5\n      path: glove/glove.hdf5\n";
        let config: SearchConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        let generator =
            ConfigSearchGenerator::new_with_datasets_dir(&config, datasets_dir.path()).unwrap();

        let mut rng = rand::rng();
        // Cursor wraps over the two queries; every query carries ground truth.
        let q0 = generator.make_query_for(0, 0, &mut rng);
        let q1 = generator.make_query_for(0, 0, &mut rng);
        let q2 = generator.make_query_for(0, 0, &mut rng);
        assert_eq!(q0.dense.as_ref().unwrap().0, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(q0.expected_ids, Some(vec![0, 2]));
        assert_eq!(q1.expected_ids, Some(vec![1, 2]));
        // Wrapped back to the first query.
        assert_eq!(q2.dense.as_ref().unwrap().0, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(q2.expected_ids, Some(vec![0, 2]));
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

    /// The generated condition matches a truncated keyword, so the prefix length
    /// is the selectivity knob.
    #[test]
    fn match_prefix_truncates_the_generated_keyword() {
        let generator = build_gen(
            "collection:\n  name: x\nrequests:\n  - kind: dense\n    size: 4\n    filters:\n      - name: color\n        type: keyword\n        source: { cardinality: 500 }\n        match_prefix: 9\n",
        );
        let mut rng = rand::rng();

        for _ in 0..20 {
            let q = generator.make_query(0, &mut rng);
            let condition = &q.filter.as_ref().unwrap().must[0];
            let Some(ConditionOneOf::Field(field)) = &condition.condition_one_of else {
                panic!("expected a field condition");
            };
            match &field.r#match.as_ref().unwrap().match_value {
                // "keyword_<n>" truncated to its first 9 characters.
                Some(MatchValue::Prefix(prefix)) => {
                    assert_eq!(prefix.chars().count(), 9, "{prefix}");
                    assert!(prefix.starts_with("keyword_"), "{prefix}");
                }
                other => panic!("expected a prefix match, got {other:?}"),
            }
        }
    }

    #[test]
    fn idf_corpus_is_generated_for_sparse_requests_only() {
        let generator = build_gen(
            "collection:\n  name: x\nrequests:\n  - kind: sparse\n    using: bm25\n    source: { vocab_size: 32, length: 4 }\n    idf_corpus:\n      - name: tenant\n        type: keyword\n        source: { cardinality: 5 }\n  - kind: dense\n    size: 4\n",
        );
        let mut rng = rand::rng();

        let sparse = generator.make_query_for(0, 0, &mut rng);
        assert!(sparse.sparse.is_some());
        assert_eq!(sparse.idf_corpus.as_ref().unwrap().must.len(), 1);
        // The corpus is separate from the query filter, which stays unset here.
        assert!(sparse.filter.is_none());

        let dense = generator.make_query_for(1, 0, &mut rng);
        assert!(dense.idf_corpus.is_none());
    }

    /// No `idf_corpus:` ⇒ no IDF params, i.e. collection-wide statistics.
    #[test]
    fn sparse_requests_have_no_idf_corpus_by_default() {
        let generator = build_gen(
            "collection:\n  name: x\nrequests:\n  - kind: sparse\n    using: bm25\n    source: { vocab_size: 32, length: 4 }\n",
        );
        let mut rng = rand::rng();
        assert!(generator.make_query(0, &mut rng).idf_corpus.is_none());
    }
}
