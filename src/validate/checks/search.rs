//! Cross-reference a search config against the upload config it runs on.

use super::cross_check::{check_collection_name, check_filter};
use crate::config::search::{SearchConfig, SearchRequestConfig};
use crate::config::vector::{SparseVectorConfig, VectorSource};
use crate::config::{ModifierKind, UploadConfig};
use crate::validate::Diagnostic;

/// Cross-reference a search config against the upload config it will run on.
pub fn check_search_against_upload(
    upload: &UploadConfig,
    search: &SearchConfig,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let c = &upload.collection;
    check_collection_name(upload, &search.collection.name, "search", &mut out);
    // A whole-payload dataset source loads fields bfb can't see, so a filter on
    // an undeclared field isn't necessarily matching nothing.
    let has_payload_source = c.payload.source.is_some();

    for (i, req) in search.requests.iter().enumerate() {
        let loc = format!("requests[{i}]");
        match req {
            SearchRequestConfig::Dense {
                using,
                size,
                datatype: _,
                source,
                filters,
            } => {
                // C1: the queried dense vector must be declared in the upload config.
                let vector = match using {
                    Some(name) => c.vectors.iter().find(|v| v.name.as_deref() == Some(name)),
                    None => c.vectors.iter().find(|v| v.name.is_none()),
                };
                let Some(vector) = vector else {
                    let message = match using {
                        // If the name exists as a sparse vector, say so — the
                        // query kind, not the name, is what's wrong.
                        Some(name) if c.sparse_vectors.iter().any(|s| &s.name == name) => format!(
                            "queries dense vector {name:?}, but the upload config declares \
                             {name:?} as a sparse vector"
                        ),
                        Some(name) => format!(
                            "queries dense vector {name:?}, which the upload config does not declare"
                        ),
                        None => "queries the unnamed default dense vector, but the upload config \
                                 declares no unnamed vector"
                            .to_string(),
                    };
                    out.push(Diagnostic::error("search", loc, message));
                    // A request whose vector is unresolved isn't worth flagging
                    // filter-by-filter; fix the vector, re-run, then filters check.
                    continue;
                };
                // C3: generated (random) queries must match the vector's dimension.
                // File / dataset query sources carry their own dimension.
                if matches!(source, VectorSource::Random) && *size != vector.size {
                    out.push(Diagnostic::error(
                        "search",
                        loc.clone(),
                        format!(
                            "dense query `size: {size}` does not match the upload dimension \
                             {dim} of vector {name}",
                            dim = vector.size,
                            name = using.as_deref().unwrap_or("(default)"),
                        ),
                    ));
                }
                // C8: random query vectors against a collection loaded from real
                // data (dataset/file) make recall/accuracy meaningless — the
                // random points have no true neighbours. Qdrant runs it fine and
                // it's valid for throughput/latency runs, so this is a warning.
                if matches!(source, VectorSource::Random) {
                    let upload_real = match &vector.source {
                        VectorSource::Dataset { .. } => Some("a dataset"),
                        VectorSource::File { .. } => Some("a file"),
                        VectorSource::Random => None,
                    };
                    if let Some(real) = upload_real {
                        let what = match using {
                            Some(n) => format!("dense vector {n:?}"),
                            None => "the default dense vector".to_string(),
                        };
                        out.push(Diagnostic::warning(
                            "search",
                            loc.clone(),
                            format!(
                                "queries {what} with random vectors, but the collection loads it \
                                 from {real} (real data) — recall/accuracy will be meaningless; \
                                 use a dataset query source for accuracy benchmarks"
                            ),
                        ));
                    }
                }
                for (j, f) in filters.iter().enumerate() {
                    check_filter(
                        &c.fields,
                        has_payload_source,
                        f,
                        "search",
                        &format!("{loc}.filters[{j}]"),
                        &mut out,
                    );
                }
            }
            SearchRequestConfig::Sparse {
                using,
                source: _,
                filters,
                idf_corpus,
            } => {
                // C2: the queried sparse vector must be declared in the upload config.
                match c.sparse_vectors.iter().find(|s| &s.name == using) {
                    None => {
                        // If the name exists as a dense vector, say so — the
                        // query kind, not the name, is what's wrong.
                        let message = if c
                            .vectors
                            .iter()
                            .any(|v| v.name.as_deref() == Some(using.as_str()))
                        {
                            format!(
                                "queries sparse vector {using:?}, but the upload config declares \
                                 {using:?} as a dense vector"
                            )
                        } else {
                            format!(
                                "queries sparse vector {using:?}, which the upload config does not \
                                 declare"
                            )
                        };
                        out.push(Diagnostic::error("search", loc.clone(), message));
                    }
                    Some(sparse) => {
                        let SparseVectorConfig {
                            name: _,
                            datatype: _,
                            on_disk: _,
                            memory: _,
                            modifier,
                            source: _,
                        } = sparse;
                        // C6: an idf_corpus needs the sparse vector's IDF modifier.
                        if !idf_corpus.is_empty() && *modifier != ModifierKind::Idf {
                            out.push(Diagnostic::error(
                                "search",
                                loc.clone(),
                                format!(
                                    "sets `idf_corpus`, but sparse vector {using:?} was not \
                                     declared with `modifier: idf`"
                                ),
                            ));
                        }
                    }
                }
                // idf_corpus conditions are payload filters too.
                for (j, f) in idf_corpus.iter().enumerate() {
                    check_filter(
                        &c.fields,
                        has_payload_source,
                        f,
                        "search",
                        &format!("{loc}.idf_corpus[{j}]"),
                        &mut out,
                    );
                }
                for (j, f) in filters.iter().enumerate() {
                    check_filter(
                        &c.fields,
                        has_payload_source,
                        f,
                        "search",
                        &format!("{loc}.filters[{j}]"),
                        &mut out,
                    );
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::checks::test_support::{errors, search, upload, warnings};

    // ---- C1: dense `using` must name a declared dense vector --------------

    #[test]
    fn dense_using_unknown_vector_is_error() {
        let up = upload("collection:\n  vectors:\n    - name: image\n      size: 8\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: missing\n    size: 8\n",
        );
        let diags = check_search_against_upload(&up, &se);
        let errs = errors(&diags);
        assert_eq!(errs.len(), 1, "{diags:?}");
        assert!(errs[0].message.contains("missing"), "{:?}", errs[0]);
    }

    #[test]
    fn dense_using_known_vector_is_ok() {
        let up = upload("collection:\n  vectors:\n    - name: image\n      size: 8\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    size: 8\n",
        );
        assert!(errors(&check_search_against_upload(&up, &se)).is_empty());
    }

    #[test]
    fn dense_default_vector_without_unnamed_upload_is_error() {
        // search omits `using` (wants the unnamed default), but upload only has
        // named vectors.
        let up = upload("collection:\n  vectors:\n    - name: image\n      size: 8\n");
        let se =
            search("collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n");
        assert_eq!(errors(&check_search_against_upload(&up, &se)).len(), 1);
    }

    #[test]
    fn dense_default_vector_with_unnamed_upload_is_ok() {
        let up = upload("collection:\n  vectors:\n    - size: 8\n");
        let se =
            search("collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n");
        assert!(errors(&check_search_against_upload(&up, &se)).is_empty());
    }

    // ---- C2: sparse `using` must name a declared sparse vector ------------

    #[test]
    fn sparse_using_unknown_vector_is_error() {
        let up =
            upload("collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: sparse\n    using: nope\n",
        );
        let errs = errors(&check_search_against_upload(&up, &se));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].message.contains("nope"));
    }

    #[test]
    fn sparse_using_known_vector_is_ok() {
        let up =
            upload("collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: sparse\n    using: bm25\n",
        );
        assert!(errors(&check_search_against_upload(&up, &se)).is_empty());
    }

    // ---- kind mismatch: a name declared as the other kind names that kind --

    #[test]
    fn sparse_query_of_a_dense_vector_names_the_kind() {
        let up = upload("collection:\n  vectors:\n    - name: a\n      size: 8\n");
        let se = search("collection:\n  name: c\nrequests:\n  - kind: sparse\n    using: a\n");
        let errs = errors(&check_search_against_upload(&up, &se));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(
            errs[0].message.contains("as a dense vector"),
            "{:?}",
            errs[0]
        );
    }

    #[test]
    fn dense_query_of_a_sparse_vector_names_the_kind() {
        let up = upload(
            "collection:\n  vectors:\n    - name: d\n      size: 8\n  sparse_vectors:\n    - name: a\n",
        );
        let se = search(
            "collection:\n  name: c\nrequests:\n  - kind: dense\n    using: a\n    size: 8\n",
        );
        let errs = errors(&check_search_against_upload(&up, &se));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(
            errs[0].message.contains("as a sparse vector"),
            "{:?}",
            errs[0]
        );
    }

    // ---- C3: dense random `size` must equal the upload vector's dim --------

    #[test]
    fn dense_random_size_mismatch_is_error() {
        let up = upload("collection:\n  vectors:\n    - name: image\n      size: 512\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    size: 8\n",
        );
        let errs = errors(&check_search_against_upload(&up, &se));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].message.contains("512") && errs[0].message.contains('8'));
    }

    #[test]
    fn dense_dataset_source_skips_size_check() {
        // A dataset query source defines its own dimension, so `size` is not
        // compared against the upload vector.
        let up = upload("collection:\n  vectors:\n    - name: image\n      size: 512\n");
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    source:\n      type: dataset\n      name: glove-25-angular\n      format: h5\n      path: glove-25-angular/glove-25-angular.hdf5\n      link: http://ann-benchmarks.com/glove-25-angular.hdf5\n",
        );
        assert!(errors(&check_search_against_upload(&up, &se)).is_empty());
    }

    // ---- C8: random query vectors against a real-data collection ----------

    const DATASET_VECTOR: &str = "collection:\n  vectors:\n    - name: image\n      size: 25\n      source:\n        type: dataset\n        name: glove-25-angular\n        format: h5\n        path: glove/glove.hdf5\n        link: http://ann-benchmarks.com/glove-25-angular.hdf5\n";

    #[test]
    fn random_search_against_dataset_upload_is_warning() {
        let up = upload(DATASET_VECTOR);
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    size: 25\n    source: random\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("random") && warns[0].message.contains("image"));
    }

    #[test]
    fn random_search_against_random_upload_is_ok() {
        // Both synthetic — no real-data mismatch to warn about.
        let up = upload(
            "collection:\n  vectors:\n    - name: image\n      size: 25\n      source: random\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    size: 25\n    source: random\n",
        );
        assert!(check_search_against_upload(&up, &se).is_empty());
    }

    #[test]
    fn dataset_search_against_dataset_upload_is_ok() {
        // The proper accuracy setup: real query vectors from a dataset — no warning.
        let up = upload(DATASET_VECTOR);
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: image\n    source:\n      type: dataset\n      name: glove-25-angular\n      format: h5\n      path: glove/glove.hdf5\n      link: http://ann-benchmarks.com/glove-25-angular.hdf5\n",
        );
        assert!(check_search_against_upload(&up, &se).is_empty());
    }

    // ---- C6: idf_corpus needs the sparse vector's modifier: idf -----------

    #[test]
    fn idf_corpus_without_idf_modifier_is_error() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n  fields:\n    - name: tenant\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: sparse\n    using: bm25\n    idf_corpus:\n      - name: tenant\n        type: keyword\n",
        );
        let errs = errors(&check_search_against_upload(&up, &se));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].message.contains("idf"));
    }

    #[test]
    fn idf_corpus_with_idf_modifier_is_ok() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n      modifier: idf\n  fields:\n    - name: tenant\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: sparse\n    using: bm25\n    idf_corpus:\n      - name: tenant\n        type: keyword\n",
        );
        assert!(errors(&check_search_against_upload(&up, &se)).is_empty());
    }
}
