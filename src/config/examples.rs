//! Built-in YAML configs, compiled into the `bfb` binary.
//!
//! These are the files under `examples/`. They stay on disk as the editable
//! source of truth; `include_str!` bakes them in so `--example name` works
//! without mounting or downloading anything.
//!
//! `bfb examples` lists them; `bfb examples <name>` prints one to stdout so it
//! can be copied and customized as a `--file`.

use std::borrow::Cow;
use std::fmt;
use std::io::{self, Write};

use anyhow::{Context, Result, bail};

/// Which subcommand a built-in config belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleKind {
    Upload,
    Search,
    Scroll,
}

impl ExampleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Search => "search",
            Self::Scroll => "scroll",
        }
    }
}

impl fmt::Display for ExampleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One shipped YAML config.
pub struct Example {
    /// CLI name: `--example upload-config`. Filename without `.yaml`.
    pub name: &'static str,
    pub kind: ExampleKind,
    pub file_name: &'static str,
    pub description: &'static str,
    pub yaml: &'static str,
}

macro_rules! example {
    ($name:literal, $kind:ident, $desc:literal) => {
        Example {
            name: $name,
            kind: ExampleKind::$kind,
            file_name: concat!($name, ".yaml"),
            description: $desc,
            yaml: include_str!(concat!("../../examples/", $name, ".yaml")),
        }
    };
}

pub static EXAMPLES: &[Example] = &[
    example!(
        "upload-config",
        Upload,
        "Full upload schema illustration (named vectors, payloads, knobs)"
    ),
    example!(
        "simple-hybrid",
        Upload,
        "Hybrid collection: dense + sparse vectors and a few payload fields"
    ),
    example!(
        "upload-dataset-config",
        Upload,
        "Upload glove-25-angular from an ann-benchmarks HDF5 dataset"
    ),
    example!(
        "upload-laion-small-clip",
        Upload,
        "Upload laion-small-clip (tar: vectors.npy + payloads.jsonl)"
    ),
    example!(
        "upload-laion-part",
        Upload,
        "One LAION-400M part: npy vectors + parquet payload"
    ),
    example!(
        "upload-laion-400m",
        Upload,
        "Full LAION-400M corpus (~410 parts, streamed with cache: evict)"
    ),
    example!(
        "serverless-upload",
        Upload,
        "Minimal upload shape for `bfb serverless upload` (dense + keyword)"
    ),
    example!(
        "search-config",
        Search,
        "Search requests matching upload-config (dense, sparse, filters)"
    ),
    example!(
        "search-dataset-accuracy",
        Search,
        "Measure recall against a dataset query set + ground truth"
    ),
    example!(
        "scroll-config",
        Scroll,
        "Scroll workload matching upload-config (filtered and unfiltered)"
    ),
];

/// A config loaded from `--file` or `--example`.
#[derive(Debug)]
pub struct ResolvedConfig {
    /// Path, or `example:<name>`, recorded in `--json` results.
    pub origin: String,
    pub yaml: Cow<'static, str>,
}

/// Strip a `.yaml` / `.yml` suffix so `--example upload-config.yaml` works.
fn normalize_name(input: &str) -> &str {
    input
        .strip_suffix(".yaml")
        .or_else(|| input.strip_suffix(".yml"))
        .unwrap_or(input)
}

pub fn lookup(input: &str) -> Option<&'static Example> {
    let name = normalize_name(input);
    EXAMPLES
        .iter()
        .find(|e| e.name == name || e.file_name == input)
}

pub fn lookup_required(input: &str) -> Result<&'static Example> {
    lookup(input).ok_or_else(|| {
        let names: Vec<_> = EXAMPLES.iter().map(|e| e.name).collect();
        anyhow::anyhow!(
            "unknown example {input:?}. Available: {}. Try `bfb examples`.",
            names.join(", ")
        )
    })
}

/// Resolve `--file` / `--example` (exactly one is set by clap).
pub fn resolve(
    file: Option<&str>,
    example: Option<&str>,
    expected: ExampleKind,
) -> Result<ResolvedConfig> {
    match (file, example) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read config file {path}"))?;
            Ok(ResolvedConfig {
                origin: path.to_string(),
                yaml: Cow::Owned(text),
            })
        }
        (None, Some(name)) => {
            let example = lookup_required(name)?;
            if example.kind != expected {
                bail!(
                    "example `{}` is a {} config; use `bfb {} --example {}`",
                    example.name,
                    example.kind,
                    example.kind,
                    example.name
                );
            }
            Ok(ResolvedConfig {
                origin: format!("example:{}", example.name),
                yaml: Cow::Borrowed(example.yaml),
            })
        }
        _ => bail!("provide exactly one of --file or --example"),
    }
}

/// Print the catalog, or dump one example's YAML to stdout.
pub fn run(name: Option<&str>) -> Result<()> {
    match name {
        None => {
            print_catalog(&mut io::stdout())?;
        }
        Some(name) => {
            let example = lookup_required(name)?;
            print!("{}", example.yaml);
            if !example.yaml.ends_with('\n') {
                println!();
            }
        }
    }
    Ok(())
}

fn print_catalog(out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "Built-in YAML configs (compiled into this binary).")?;
    writeln!(out, "Use `--example <name>` instead of `--file`:")?;
    writeln!(
        out,
        "  bfb upload --example upload-config -n 1M --uri http://localhost:6334"
    )?;
    writeln!(out, "Print one to customize it:")?;
    writeln!(out, "  bfb examples upload-config > my.yaml")?;
    writeln!(out)?;

    for kind in [
        ExampleKind::Upload,
        ExampleKind::Search,
        ExampleKind::Scroll,
    ] {
        writeln!(out, "{kind}")?;
        for example in EXAMPLES.iter().filter(|e| e.kind == kind) {
            writeln!(out, "  {:<28} {}", example.name, example.description)?;
        }
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_entry_has_a_unique_name() {
        let mut names: Vec<_> = EXAMPLES.iter().map(|e| e.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), EXAMPLES.len());
    }

    #[test]
    fn lookup_accepts_bare_name_and_filename() {
        let by_name = lookup("upload-config").unwrap();
        let by_file = lookup("upload-config.yaml").unwrap();
        assert_eq!(by_name.name, by_file.name);
        assert_eq!(by_name.kind, ExampleKind::Upload);
    }

    #[test]
    fn every_embedded_example_parses() {
        for example in EXAMPLES {
            match example.kind {
                ExampleKind::Upload => {
                    crate::config::parse(example.yaml, example.name).unwrap();
                }
                ExampleKind::Search => {
                    crate::config::search::parse(example.yaml, example.name).unwrap();
                }
                ExampleKind::Scroll => {
                    crate::config::scroll::parse(example.yaml, example.name).unwrap();
                }
            }
        }
    }

    #[test]
    fn catalog_covers_examples_directory() {
        let dir = std::path::Path::new("examples");
        if !dir.is_dir() {
            // `cargo test --offline` from another cwd still compiles the
            // `include_str!` payloads; skip the on-disk coverage check.
            return;
        }
        let mut on_disk = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".yaml") {
                on_disk.push(name.into_owned());
            }
        }
        on_disk.sort();
        let mut catalog: Vec<_> = EXAMPLES.iter().map(|e| e.file_name.to_string()).collect();
        catalog.sort();
        assert_eq!(
            on_disk, catalog,
            "every file in examples/ must be listed in EXAMPLES (and vice versa)"
        );
    }

    #[test]
    fn reject_wrong_kind() {
        let err = resolve(None, Some("search-config"), ExampleKind::Upload)
            .unwrap_err()
            .to_string();
        assert!(err.contains("search"), "{err}");
        assert!(err.contains("bfb search --example search-config"), "{err}");
    }

    #[test]
    fn resolve_reads_a_custom_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom.yaml");
        std::fs::write(&path, lookup("simple-hybrid").unwrap().yaml).unwrap();
        let resolved = resolve(Some(path.to_str().unwrap()), None, ExampleKind::Upload).unwrap();
        assert_eq!(resolved.origin, path.to_str().unwrap());
        crate::config::parse(&resolved.yaml, &resolved.origin).unwrap();
    }

    #[test]
    fn catalog_text_lists_names_and_usage() {
        let mut buf = Vec::new();
        print_catalog(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("--example"));
        assert!(text.contains("upload-config"));
        assert!(text.contains("search-config"));
        assert!(text.contains("scroll-config"));
    }
}
