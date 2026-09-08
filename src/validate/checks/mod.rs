//! Pure config-consistency checks. No I/O, no Qdrant — every function takes
//! already-parsed configs and returns a list of [`Diagnostic`]s, so each can be
//! unit-tested in isolation.
//!
//! Split by role: [`upload`] lints a single upload config; [`search`] and
//! [`scroll`] cross-reference a search / scroll config against the upload it
//! runs on; [`cross_check`] holds the shared payload-filter check both of them
//! call.
//!
//! [`Diagnostic`]: crate::validate::Diagnostic

pub mod cross_check;
pub mod scroll;
pub mod search;
pub mod upload;

pub use scroll::check_scroll_against_upload;
pub use search::check_search_against_upload;
pub use upload::lint_upload;

/// Shared test helpers: parse a config from YAML and split diagnostics by
/// severity. Kept here so every role's test module reuses the same setup.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::config::UploadConfig;
    use crate::config::scroll::ScrollConfig;
    use crate::config::search::SearchConfig;
    use crate::validate::{Diagnostic, Severity};

    pub fn upload(yaml: &str) -> UploadConfig {
        crate::config::parse(yaml, "test-upload").unwrap()
    }
    pub fn search(yaml: &str) -> SearchConfig {
        crate::config::search::parse(yaml, "test-search").unwrap()
    }
    pub fn scroll(yaml: &str) -> ScrollConfig {
        crate::config::scroll::parse(yaml, "test-scroll").unwrap()
    }

    pub fn warnings(diags: &[Diagnostic]) -> Vec<Diagnostic> {
        diags
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .cloned()
            .collect()
    }

    pub fn errors(diags: &[Diagnostic]) -> Vec<Diagnostic> {
        diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .cloned()
            .collect()
    }
}
