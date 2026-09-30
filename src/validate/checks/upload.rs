//! Single-file lints for an upload config.

use crate::config::vector::{SparseVectorConfig, VectorConfig, VectorSource};
use crate::config::{DatatypeKind, UploadConfig};
use crate::validate::Diagnostic;

/// Single-file lints for an upload config that go beyond `UploadConfig::validate()`.
/// Both are errors because Qdrant rejects the config outright: a declared
/// dimension that contradicts the dataset's `vector_size` (L2), and the `turbo4`
/// datatype on a sparse vector. Verified against the server source; we don't
/// encode server behavior we couldn't confirm as a hard rejection.
pub fn lint_upload(upload: &UploadConfig) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    for (i, vector) in upload.collection.vectors.iter().enumerate() {
        let VectorConfig {
            name: _,
            size,
            distance: _,
            datatype: _,
            on_disk: _,
            memory: _,
            multivector: _,
            quantization: _,
            source,
        } = vector;

        match source {
            // L2: a declared dimension that contradicts the dataset's own
            // `vector_size` cannot possibly load — a logical inconsistency, not
            // a server rule.
            VectorSource::Dataset { dataset } => {
                if let Some(ds_dim) = dataset.vector_size
                    && ds_dim != *size
                {
                    out.push(Diagnostic::error(
                        "upload",
                        format!("vectors[{i}]"),
                        format!(
                            "declares `size: {size}` but its dataset source declares \
                             `vector_size: {ds_dim}`"
                        ),
                    ));
                }
            }
            VectorSource::Random | VectorSource::File { .. } => {}
        }
    }

    for (i, sparse) in upload.collection.sparse_vectors.iter().enumerate() {
        let SparseVectorConfig {
            name,
            datatype,
            on_disk: _,
            memory: _,
            modifier: _,
            source: _,
        } = sparse;

        // Qdrant rejects the `turbo4` datatype on a sparse vector at collection
        // creation (`unsupported_sparse_datatype`): turbo4 is a dense-only
        // turbo-quantized storage and sparse vectors do not support it.
        if *datatype == DatatypeKind::Turbo4 {
            out.push(Diagnostic::error(
                "upload",
                format!("sparse_vectors[{i}]"),
                format!(
                    "sparse vector {name:?} sets `datatype: turbo4`, which Qdrant does not \
                     support for sparse vectors (turbo4 is a dense-only quantized storage)"
                ),
            ));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::checks::test_support::{errors, upload};

    // ---- turbo4 datatype on a sparse vector: Qdrant rejects it ------------

    #[test]
    fn sparse_turbo4_datatype_is_error() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n      datatype: turbo4\n",
        );
        let errs = errors(&lint_upload(&up));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].message.contains("turbo4") && errs[0].message.contains("bm25"));
    }

    #[test]
    fn sparse_supported_datatype_is_ok() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  sparse_vectors:\n    - name: bm25\n      datatype: float16\n",
        );
        assert!(lint_upload(&up).is_empty());
    }

    // `lint_upload` does not lint `memory: pinned`: unlike the turbo4-on-sparse
    // rejection above (a verified, dedicated server error), we did not confirm
    // the server rejects a pinned dense/payload placement — it may be silently
    // ignored — so encoding it would be an unverified server rule. No diagnostic.

    #[test]
    fn pinned_placements_are_not_linted() {
        let dense = upload("collection:\n  vectors:\n    - size: 8\n      memory: pinned\n");
        let payload =
            upload("collection:\n  vectors:\n    - size: 8\n  payload:\n    memory: pinned\n");
        assert!(lint_upload(&dense).is_empty());
        assert!(lint_upload(&payload).is_empty());
    }

    // ---- L2: declared vector size must match the dataset's vector_size -----

    #[test]
    fn vector_size_mismatch_with_dataset_is_error() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 512\n      source:\n        type: dataset\n        name: glove-25-angular\n        format: h5\n        path: glove-25-angular/glove-25-angular.hdf5\n        link: http://ann-benchmarks.com/glove-25-angular.hdf5\n        vector_size: 25\n",
        );
        let errs = errors(&lint_upload(&up));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].message.contains("512") && errs[0].message.contains("25"));
    }

    #[test]
    fn vector_size_matching_dataset_is_ok() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 25\n      source:\n        type: dataset\n        name: glove-25-angular\n        format: h5\n        path: glove-25-angular/glove-25-angular.hdf5\n        link: http://ann-benchmarks.com/glove-25-angular.hdf5\n        vector_size: 25\n",
        );
        assert!(lint_upload(&up).is_empty());
    }

    #[test]
    fn dataset_without_vector_size_is_skipped() {
        // No inline `vector_size` (it would come from a registry) ⇒ nothing to
        // compare, so no diagnostic.
        let up = upload(
            "collection:\n  vectors:\n    - size: 512\n      source:\n        type: dataset\n        name: glove-25-angular\n        format: h5\n        path: glove-25-angular/glove-25-angular.hdf5\n        link: http://ann-benchmarks.com/glove-25-angular.hdf5\n",
        );
        assert!(lint_upload(&up).is_empty());
    }
}
