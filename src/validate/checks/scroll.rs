//! Cross-reference a scroll config against the upload config it runs on.

use super::cross_check::check_filter;
use crate::config::UploadConfig;
use crate::config::scroll::{ScrollConfig, ScrollRequestConfig};
use crate::validate::Diagnostic;

/// Cross-reference a scroll config against the upload config it will run on.
/// Scroll requests carry only payload filters, so this is the filter check
/// (C4 / C5 / C7) applied to each condition.
pub fn check_scroll_against_upload(
    upload: &UploadConfig,
    scroll: &ScrollConfig,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let fields = &upload.collection.fields;
    let has_payload_source = upload.collection.payload.source.is_some();
    for (i, req) in scroll.requests.iter().enumerate() {
        let ScrollRequestConfig { filters } = req;
        for (j, f) in filters.iter().enumerate() {
            check_filter(
                fields,
                has_payload_source,
                f,
                "scroll",
                &format!("requests[{i}].filters[{j}]"),
                &mut out,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::checks::test_support::{errors, scroll, upload, warnings};

    #[test]
    fn scroll_filter_field_not_declared_is_warning() {
        let up = upload(
            "collection:\n  vectors:\n    - size: 8\n  fields:\n    - name: color\n      type: keyword\n",
        );
        let sc = scroll(
            "collection:\n  name: benchmark\nrequests:\n  - filters:\n      - name: shape\n        type: keyword\n",
        );
        let diags = check_scroll_against_upload(&up, &sc);
        assert!(errors(&diags).is_empty(), "{diags:?}");
        let warns = warnings(&diags);
        assert_eq!(warns.len(), 1, "{warns:?}");
        assert_eq!(warns[0].role, "scroll");
        assert!(warns[0].message.contains("shape"));
    }
}
