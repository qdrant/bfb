//! Shared payload-filter check, used by both the search and scroll
//! cross-references.

use crate::config::payload::PayloadConfig;
use crate::config::search::FilterPayloadConfig;
use crate::config::{PayloadType, UploadConfig};
use crate::validate::Diagnostic;

/// Warn when the search / scroll config targets a different collection than the
/// upload config creates. When both files are validated together they are meant
/// as a pair, so a differing name is almost always a typo that would silently
/// run against the wrong (or an empty) collection. It's a **warning**, not an
/// error, because searching a separate, pre-existing collection is occasionally
/// intentional.
pub(super) fn check_collection_name(
    upload: &UploadConfig,
    target_name: &str,
    role: &'static str,
    out: &mut Vec<Diagnostic>,
) {
    if upload.collection.name != target_name {
        out.push(Diagnostic::warning(
            role,
            "collection.name",
            format!(
                "targets collection {target_name:?}, but the upload config creates \
                 {upload_name:?} — the {role} will run against a different collection than the \
                 upload populates (a name typo silently yields empty results / recall 0)",
                upload_name = upload.collection.name,
            ),
        ));
    }
}

/// Check one payload filter condition against the upload config's declared
/// fields. Every problem here is a **warning**, not an error: Qdrant tolerates
/// all of them (verified against the server source). They flag a filter that
/// will silently match nothing (C4 / C9) or run unaccelerated (C5 / C7) — a
/// mismeasured benchmark, not a failed request.
pub(super) fn check_filter(
    fields: &[PayloadConfig],
    has_whole_payload_source: bool,
    filter: &FilterPayloadConfig,
    role: &'static str,
    loc: &str,
    out: &mut Vec<Diagnostic>,
) {
    let FilterPayloadConfig {
        name,
        kind,
        source: _,
        match_any: _,
        match_prefix,
    } = filter;

    let Some(field) = fields.iter().find(|f| f.name == *name) else {
        // C4: an undeclared field. With no whole-payload source, payloads come
        // only from `fields`, so the field is genuinely absent and Qdrant
        // (which treats an absent field as non-matching) returns nothing. But a
        // `payload.source` dataset loads the whole payload object, so the field
        // may be present-but-unindexed — bfb can't see the dataset schema, so
        // don't claim it matches nothing.
        let message = if has_whole_payload_source {
            format!(
                "filters on field {name:?}, which is not declared as a field; the whole-payload \
                 dataset source may still provide it (running unindexed), or not (then the filter \
                 matches nothing)"
            )
        } else {
            format!(
                "filters on field {name:?}, which the upload config does not declare — the filter \
                 will match no points"
            )
        };
        out.push(Diagnostic::warning(role, loc, message));
        return;
    };

    let PayloadConfig {
        name: _,
        kind: field_kind,
        index,
        on_disk: _,
        memory: _,
        is_tenant: _,
        is_principal: _,
        range_index: _,
        prefix,
        tokenizer: _,
        source: _,
    } = field;

    if field_kind != kind {
        // C4: a type-mismatched condition never matches the declared type
        // (Qdrant returns `false` rather than rejecting). Certain when bfb
        // generates the values from this declaration; under a whole-payload
        // source the dataset's actual value type is unknown, so don't assert it.
        let consequence = if has_whole_payload_source {
            "; whether it matches depends on the dataset payload's value type"
        } else {
            " — the filter will match no points"
        };
        out.push(Diagnostic::warning(
            role,
            loc,
            format!(
                "filters on field {name:?} as `{kind:?}`, but the upload config declares it as \
                 `{field_kind:?}`{consequence}"
            ),
        ));
        return;
    }

    // C9: config-driven UUID filters generate a fresh random UUID per query
    // (`generators/queries.rs`), which won't collide with the random UUIDs
    // stored at upload — there is no bounded-cardinality space as with keywords.
    // So the filter matches nothing. Qdrant runs it fine, hence a warning.
    if *kind == PayloadType::Uuid {
        out.push(Diagnostic::warning(
            role,
            loc,
            format!(
                "filters on UUID field {name:?} — bfb generates a random UUID per query, which \
                 won't match any stored UUID, so the filter matches nothing"
            ),
        ));
    }

    if match_prefix.is_some() && !*prefix {
        // C5: prefix matching still returns correct results, but cannot use the
        // prefix index (full scan) unless it was built with `prefix: true`.
        out.push(Diagnostic::warning(
            role,
            loc,
            format!(
                "filters on field {name:?} with `match_prefix`, but its keyword index was not \
                 declared with `prefix: true` — the query works but cannot use the prefix index \
                 (full scan)"
            ),
        ));
    }

    if !*index {
        out.push(Diagnostic::warning(
            role,
            loc,
            format!(
                "filters on field {name:?}, which is declared `index: false` (queries will full-scan)"
            ),
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::validate::checks::search::check_search_against_upload;
    use crate::validate::checks::test_support::{errors, search, upload, warnings};

    // The filter check is exercised through `check_search_against_upload` (its
    // caller), which is how it runs in practice; the scroll path has its own
    // test in `scroll.rs`.

    // ---- C4: filter field mismatch is a warning (Qdrant tolerates it — the
    //          filter just matches nothing; verified in the server source) -----

    #[test]
    fn filter_field_not_declared_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: shape\n        type: keyword\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("shape"));
    }

    #[test]
    fn filter_field_type_mismatch_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: integer\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("color"));
    }

    #[test]
    fn filter_field_matching_type_is_ok() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(
            errors(&diags).is_empty() && warnings(&diags).is_empty(),
            "{diags:?}"
        );
    }

    #[test]
    fn missing_filter_field_with_whole_payload_source_does_not_claim_no_match() {
        // The whole payload comes from a dataset, so an undeclared field may
        // still be present (unindexed) — the linter must not assert it matches
        // nothing.
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  payload:\n    source:\n      type: dataset\n      dataset:\n        name: d\n        format: tar\n        path: d/d\n        link: https://example.com/d.tgz\n  fields:\n    - name: similarity\n      type: float\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: url\n        type: keyword\n",
        );
        let warns = warnings(&check_search_against_upload(&up, &se));
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("url"), "{:?}", warns[0]);
        assert!(
            !warns[0].message.contains("match no points"),
            "{:?}",
            warns[0]
        );
    }

    #[test]
    fn missing_filter_field_without_payload_source_claims_no_match() {
        // No whole-payload source: payloads come only from `fields`, so an
        // undeclared field is genuinely absent and the filter matches nothing.
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: shape\n        type: keyword\n",
        );
        let warns = warnings(&check_search_against_upload(&up, &se));
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(
            warns[0].message.contains("match no points"),
            "{:?}",
            warns[0]
        );
    }

    // ---- C5: match_prefix without prefix index is a warning (Qdrant runs it
    //          correctly, just unaccelerated; verified in the server source) ----

    #[test]
    fn match_prefix_without_prefix_index_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n        match_prefix: 3\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("prefix"));
    }

    #[test]
    fn match_prefix_with_prefix_index_is_ok() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n      prefix: true\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n        match_prefix: 3\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(
            errors(&diags).is_empty() && warnings(&diags).is_empty(),
            "{diags:?}"
        );
    }

    // ---- C7: filtering an unindexed field is a warning, not an error ------

    #[test]
    fn filter_on_unindexed_field_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n      index: false\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(warns[0].message.contains("color"));
    }

    // ---- C9: a UUID filter generates random UUIDs that never match ---------

    #[test]
    fn uuid_filter_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: id\n      type: uuid\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: id\n        type: uuid\n",
        );
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert!(
            warns[0].message.to_lowercase().contains("uuid"),
            "{:?}",
            warns[0]
        );
    }

    #[test]
    fn non_uuid_indexed_filter_has_no_uuid_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let se = search(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    size: 8\n    filters:\n      - name: color\n        type: keyword\n",
        );
        assert!(check_search_against_upload(&up, &se).is_empty());
    }

    // ---- collection-name mismatch between the upload and the search config --

    #[test]
    fn collection_name_mismatch_is_warning() {
        let up = upload("collection:\n  name: uploaded\n  vectors:\n    - size: 8\n");
        let se = search("collection:\n  name: searched\nrequests:\n  - kind: dense\n    size: 8\n");
        let diags = check_search_against_upload(&up, &se);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert_eq!(warns[0].location, "collection.name");
        assert!(
            warns[0].message.contains("uploaded") && warns[0].message.contains("searched"),
            "{:?}",
            warns[0]
        );
    }

    #[test]
    fn matching_collection_name_is_ok() {
        let up = upload("collection:\n  name: same\n  vectors:\n    - size: 8\n");
        let se = search("collection:\n  name: same\nrequests:\n  - kind: dense\n    size: 8\n");
        assert!(check_search_against_upload(&up, &se).is_empty());
    }
}
