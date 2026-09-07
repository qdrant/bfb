//! `bfb validate`: offline, server-free checking of upload / search / scroll
//! YAML configs.
//!
//! Each role (`--upload` / `--search` / `--scroll`) is parsed through its
//! existing `config::…::parse` (so all of today's per-file `validate()` runs),
//! and — when an upload config is supplied alongside a search or scroll config —
//! [`checks`] cross-references the two to catch mismatches that are invisible
//! from a single file (a queried vector name that was never declared, a
//! `match_prefix` filter on a field not built with `prefix: true`, …).
//!
//! Nothing here connects to Qdrant: it is pure YAML-against-YAML analysis.

pub mod checks;

/// Severity of a single [`Diagnostic`]. Only an [`Severity::Error`] makes
/// `bfb validate` exit non-zero; warnings are advisory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One problem found in a config (or between two configs).
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Which config the problem is reported against: `"upload"`, `"search"`,
    /// or `"scroll"`.
    pub role: &'static str,
    /// Where in that config, e.g. `"vectors[1]"` or `"requests[0].filters[0]"`.
    pub location: String,
    pub message: String,
}

impl Diagnostic {
    pub fn error(role: &'static str, location: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            role,
            location: location.into(),
            message: message.into(),
        }
    }

    pub fn warning(
        role: &'static str,
        location: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            role,
            location: location.into(),
            message: message.into(),
        }
    }
}

use anyhow::Result;

use crate::args::{ValidateArgs, ValidateFormat};

/// `bfb validate`: read every provided role, check each, cross-check the pair(s),
/// print the result, and report whether validation passed (so `main` can pick
/// the exit code). Never returns `Err` for a config problem — those are
/// diagnostics; `Err` is reserved for genuinely unexpected failures.
pub fn run(args: &ValidateArgs) -> Result<bool> {
    let diagnostics = collect(args);

    // Cross-checks need the upload config; say so rather than silently skipping.
    if args.format == ValidateFormat::Human && args.upload.is_none() {
        if args.search.is_some() {
            println!("note: no --upload given; skipping search↔upload cross-checks");
        }
        if args.scroll.is_some() {
            println!("note: no --upload given; skipping scroll↔upload cross-checks");
        }
    }

    let output = match args.format {
        ValidateFormat::Human => render_human(&diagnostics),
        ValidateFormat::Json => render_json(&diagnostics),
    };
    print!("{output}");

    Ok(!has_errors(&diagnostics))
}

fn has_errors(diagnostics: &[Diagnostic]) -> bool {
    diagnostics.iter().any(|d| d.severity == Severity::Error)
}

/// Read and parse each provided role, then run the single-file and cross-config
/// checks. Parse / read failures become error diagnostics (so all roles are
/// still reported) rather than aborting.
pub fn collect(args: &ValidateArgs) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    let upload = load(args.upload.as_deref(), "upload", crate::config::parse, &mut diagnostics);
    let search = load(
        args.search.as_deref(),
        "search",
        crate::config::search::parse,
        &mut diagnostics,
    );
    let scroll = load(
        args.scroll.as_deref(),
        "scroll",
        crate::config::scroll::parse,
        &mut diagnostics,
    );

    if let Some(upload) = &upload {
        diagnostics.extend(checks::lint_upload(upload));
        if let Some(search) = &search {
            diagnostics.extend(checks::check_search_against_upload(upload, search));
        }
        if let Some(scroll) = &scroll {
            diagnostics.extend(checks::check_scroll_against_upload(upload, scroll));
        }
    }

    diagnostics
}

/// Read a role's file and parse it, turning a read or parse failure into an
/// error [`Diagnostic`] (and `None`) instead of aborting the whole run.
fn load<T>(
    path: Option<&str>,
    role: &'static str,
    parse: impl Fn(&str, &str) -> Result<T>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<T> {
    let path = path?;
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => {
            diagnostics.push(Diagnostic::error(role, path.to_string(), format!("cannot read file: {e}")));
            return None;
        }
    };
    match parse(&text, path) {
        Ok(config) => Some(config),
        Err(e) => {
            // `{:#}` renders anyhow's context chain on one line.
            diagnostics.push(Diagnostic::error(role, path.to_string(), format!("{e:#}")));
            None
        }
    }
}

fn severity_str(severity: Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
    }
}

fn render_human(diagnostics: &[Diagnostic]) -> String {
    let mut out = String::new();
    for d in diagnostics {
        let mark = match d.severity {
            Severity::Error => '✗',
            Severity::Warning => '⚠',
        };
        out.push_str(&format!(
            "{mark} [{}] {} {}: {}\n",
            severity_str(d.severity),
            d.role,
            d.location,
            d.message
        ));
    }

    let errors = diagnostics.iter().filter(|d| d.severity == Severity::Error).count();
    let warnings = diagnostics.len() - errors;
    if diagnostics.is_empty() {
        out.push_str("✓ no problems found\n");
    } else {
        out.push_str(&format!("\n{errors} error(s), {warnings} warning(s)\n"));
    }
    out
}

fn render_json(diagnostics: &[Diagnostic]) -> String {
    let items: Vec<serde_json::Value> = diagnostics
        .iter()
        .map(|d| {
            serde_json::json!({
                "severity": severity_str(d.severity),
                "role": d.role,
                "location": d.location,
                "message": d.message,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "ok": !has_errors(diagnostics),
        "diagnostics": items,
    });
    let mut out = serde_json::to_string_pretty(&doc).expect("diagnostics serialize");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_config(text: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(text.as_bytes()).unwrap();
        f
    }

    fn args_for(upload: Option<&str>, search: Option<&str>) -> ValidateArgs {
        ValidateArgs {
            upload: upload.map(str::to_string),
            search: search.map(str::to_string),
            scroll: None,
            format: ValidateFormat::Human,
        }
    }

    #[test]
    fn render_json_reports_ok_false_on_error() {
        let diags = vec![Diagnostic::error("search", "requests[0]", "bad")];
        let value: serde_json::Value = serde_json::from_str(&render_json(&diags)).unwrap();
        assert_eq!(value["ok"], serde_json::json!(false));
        assert_eq!(value["diagnostics"].as_array().unwrap().len(), 1);
        assert_eq!(value["diagnostics"][0]["severity"], "error");
    }

    #[test]
    fn render_json_ok_true_when_only_warnings() {
        let diags = vec![Diagnostic::warning("upload", "fields[0]", "meh")];
        let value: serde_json::Value = serde_json::from_str(&render_json(&diags)).unwrap();
        assert_eq!(value["ok"], serde_json::json!(true));
    }

    #[test]
    fn render_human_marks_errors() {
        let diags = vec![Diagnostic::error("search", "requests[0]", "boom")];
        let text = render_human(&diags);
        assert!(text.contains("boom") && text.contains("requests[0]"), "{text}");
        assert!(text.contains("error"), "{text}");
    }

    #[test]
    fn render_human_reports_clean() {
        assert!(render_human(&[]).contains('✓'));
    }

    #[test]
    fn collect_lints_upload_file() {
        // L2: declared size contradicts the dataset's own vector_size.
        let up = write_config(
            "collection:\n  vectors:\n    - size: 512\n      source:\n        type: dataset\n        name: glove-25-angular\n        format: h5\n        path: glove-25-angular/glove.hdf5\n        link: http://ann-benchmarks.com/glove-25-angular.hdf5\n        vector_size: 25\n",
        );
        let diags = collect(&args_for(up.path().to_str(), None));
        assert!(has_errors(&diags), "{diags:?}");
    }

    #[test]
    fn collect_cross_checks_search_against_upload() {
        let up = write_config("collection:\n  vectors:\n    - name: image\n      size: 8\n");
        let se = write_config(
            "collection:\n  name: benchmark\nrequests:\n  - kind: dense\n    using: missing\n    size: 8\n",
        );
        let diags = collect(&args_for(up.path().to_str(), se.path().to_str()));
        assert!(
            diags.iter().any(|d| d.severity == Severity::Error && d.message.contains("missing")),
            "{diags:?}"
        );
    }

    #[test]
    fn collect_reports_unreadable_file() {
        let diags = collect(&args_for(Some("/no/such/bfb-config.yaml"), None));
        assert!(has_errors(&diags), "{diags:?}");
    }
}
