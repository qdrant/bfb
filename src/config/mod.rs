//! YAML collection-shape configuration for `bfb upload --file` / `--example`.
//!
//! The config file describes only the *shape* of the data (collection params +
//! how each field's values are generated). The *how* of uploading (number of
//! points, batch size, threads, parallelism, uri, …) stays on the CLI.
//!
//! See `BENCHMARK_ROADMAP.md` §2 for the schema rationale and `examples/`.

pub mod collection;
pub mod examples;
pub mod payload;
pub mod schema;
pub mod scroll;
pub mod search;
pub mod vector;

pub use collection::{CollectionConfig, IdType, MemoryKind, QuantKind};
pub use payload::{PayloadSource, PayloadSourceKind, PayloadType, TokenizerKind};
pub use vector::{
    ComparatorKind, DatatypeKind, DistanceKind, DistributionKind, FileStrategy, ModifierKind,
    SparseKind, SparseSource, VectorConfig, VectorSource,
};

use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// Top-level document: `{ collection: { ... } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadConfig {
    pub collection: CollectionConfig,
}

// ------------------------------ Loading / validation ---------------------

/// Parse and validate a YAML upload config.
pub fn parse(text: &str, origin: &str) -> Result<UploadConfig> {
    let config: UploadConfig =
        serde_yaml::from_str(text).with_context(|| format!("failed to parse config {origin}"))?;
    config.validate()?;
    Ok(config)
}

impl UploadConfig {
    pub fn validate(&self) -> Result<()> {
        let c = &self.collection;

        if c.vectors.is_empty() && c.sparse_vectors.is_empty() {
            bail!("config must define at least one dense or sparse vector");
        }

        // Dense vector names: at most one unnamed; the rest must be named & unique.
        let unnamed = c.vectors.iter().filter(|v| v.name.is_none()).count();
        if unnamed > 1 {
            bail!("at most one dense vector may omit `name` (the default vector)");
        }
        if unnamed == 1 && c.vectors.len() > 1 {
            // The unnamed default vector cannot coexist with named ones in a map.
            bail!("when multiple dense vectors are defined, every vector must have a `name`");
        }

        let mut names = std::collections::HashSet::new();
        for v in &c.vectors {
            if let Some(n) = &v.name
                && !names.insert(n.clone())
            {
                bail!("duplicate vector name {n:?}");
            }
            match &v.source {
                VectorSource::File { path, .. } => validate_file_source_path(path)?,
                VectorSource::Dataset { dataset } => dataset.validate_inline()?,
                _ => {}
            }
            if let Some(mv) = &v.multivector
                && mv.count == 0
            {
                bail!("multivector.count must be > 0");
            }
        }
        for s in &c.sparse_vectors {
            if !names.insert(s.name.clone()) {
                bail!(
                    "sparse vector name {:?} collides with another vector",
                    s.name
                );
            }
            match s.source.kind {
                SparseKind::Random => {
                    if s.source.length == 0 {
                        bail!("sparse vector {:?}: length must be > 0", s.name);
                    }
                    if s.source.length > s.source.vocab_size {
                        bail!(
                            "sparse vector {:?}: length ({}) must be <= vocab_size ({})",
                            s.name,
                            s.source.length,
                            s.source.vocab_size
                        );
                    }
                }
                SparseKind::Dataset => {
                    let dataset = s
                        .source
                        .dataset
                        .as_ref()
                        .context("sparse dataset source is missing dataset fields")?;
                    dataset.validate_inline()?;
                }
            }
        }

        // Whole-payload source (`payload.source`): only a dataset makes sense, and
        // it provides the whole object (no per-field `field`).
        if let Some(src) = &c.payload.source {
            if src.kind != PayloadSourceKind::Dataset {
                bail!("`payload.source` must be `type: dataset`");
            }
            let dataset = src
                .dataset
                .as_ref()
                .context("`payload.source` is missing dataset fields")?;
            dataset.validate_inline()?;
            if src.field.is_some() {
                bail!("`payload.source` must not set `field` (it loads the whole payload)");
            }
        }

        let mut payload_names = std::collections::HashSet::new();
        for p in &c.fields {
            if !payload_names.insert(p.name.clone()) {
                bail!("duplicate payload field name {:?}", p.name);
            }
            let Some(src) = &p.source else {
                // No per-field source: values come from `payload.source` (or,
                // absent that, are randomly generated). Nothing to validate.
                continue;
            };
            if let Some(c) = src.clusters
                && c == 0
            {
                bail!("payload {:?}: clusters must be > 0", p.name);
            }
            if src.kind == PayloadSourceKind::Dataset {
                let dataset = src
                    .dataset
                    .as_ref()
                    .context("payload dataset source is missing dataset fields")?;
                dataset.validate_inline()?;
                if src.field.as_deref().unwrap_or("").is_empty() {
                    bail!(
                        "payload {:?}: dataset source requires `field` (schema field name)",
                        p.name
                    );
                }
            }
        }

        if let Some(sh) = &c.sharding
            && sh.method != "custom"
        {
            bail!(
                "only `custom` sharding method is supported, got {:?}",
                sh.method
            );
        }

        Ok(())
    }
}

/// Validate a `source: {type: file}` path. `http(s)://` URLs are fetched and
/// cached on first use; local paths must already exist. Other URL schemes are
/// rejected up front rather than failing when the file is opened.
pub(crate) fn validate_file_source_path(path: &str) -> Result<()> {
    if crate::dataset::is_remote_url(path) {
        return Ok(());
    }
    if let Some((scheme, _)) = path.split_once("://") {
        bail!(
            "vector source path uses unsupported scheme {scheme:?}: {path}\n\
             `type: file` accepts a local path or an http(s):// URL. \
             For data hosted on S3, use its https:// URL, or a `type: dataset` source."
        );
    }
    if !Path::new(path).exists() {
        bail!("vector source file not found: {path}");
    }
    Ok(())
}

// --------------------------- serde helpers -------------------------------

/// Visitor for a value that may be either a bare string (shorthand, parsed via
/// `FromStr`) or a full map (parsed via `Deserialize`). Standard serde idiom.
struct StringOrStruct<T>(PhantomData<fn() -> T>);

impl<'de, T> Visitor<'de> for StringOrStruct<T>
where
    T: Deserialize<'de> + FromStr<Err = String>,
{
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string or a map")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<T, E> {
        FromStr::from_str(value).map_err(de::Error::custom)
    }

    fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
        Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))
    }
}

/// Deserialize a string-or-map value (see [`StringOrStruct`]).
pub(crate) fn string_or_struct<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = String>,
    D: Deserializer<'de>,
{
    deserializer.deserialize_any(StringOrStruct(PhantomData))
}

/// Like [`string_or_struct`], but for an optional field: `null`/absent ⇒ `None`,
/// otherwise a string or map ⇒ `Some`.
pub(crate) fn option_string_or_struct<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = String>,
    D: Deserializer<'de>,
{
    struct OptVisitor<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for OptVisitor<T>
    where
        T: Deserialize<'de> + FromStr<Err = String>,
    {
        type Value = Option<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, a string, or a map")
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<T>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<T>, E> {
            Ok(None)
        }

        fn visit_some<D2: Deserializer<'de>>(self, d: D2) -> Result<Option<T>, D2::Error> {
            d.deserialize_any(StringOrStruct(PhantomData)).map(Some)
        }
    }

    deserializer.deserialize_option(OptVisitor(PhantomData))
}

fn default_true() -> bool {
    true
}
fn default_one() -> u32 {
    1
}
pub(crate) fn default_collection_name() -> String {
    "benchmark".to_string()
}
fn default_custom() -> String {
    "custom".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bfb` echoes the parsed config back to the console, which serializes it.
    /// A dataset source flattens an inline dataset definition into an
    /// internally-tagged enum variant — the one shape serde can refuse to
    /// serialize — so keep a config that uses it round-trippable.
    #[test]
    fn effective_config_renders_back_to_yaml() {
        let yaml = r#"
collection:
  name: glove
  vectors:
    - size: 25
      distance: cosine
      source:
        type: dataset
        name: glove-25-angular
        format: h5
        path: glove-25-angular/glove-25-angular.hdf5
  fields:
    - name: a
      type: keyword
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();

        let rendered = serde_yaml::to_string(&cfg).expect("config must be printable");
        assert!(rendered.contains("glove-25-angular"), "{rendered}");
        // Defaults are materialized, which is the point of printing it.
        assert!(rendered.contains("datatype: float32"), "{rendered}");

        let reparsed: UploadConfig = serde_yaml::from_str(&rendered).unwrap();
        reparsed.validate().unwrap();
    }

    #[test]
    fn parses_minimal_config() {
        let yaml = r#"
collection:
  name: test
  vectors:
    - size: 128
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.collection.name, "test");
        assert_eq!(cfg.collection.vectors.len(), 1);
        assert_eq!(cfg.collection.vectors[0].size, 128);
        assert!(matches!(
            cfg.collection.vectors[0].source,
            VectorSource::Random
        ));
        assert!(matches!(cfg.collection.id, IdType::Integer));
    }

    #[test]
    fn source_shorthand_and_map() {
        let yaml = r#"
collection:
  vectors:
    - name: a
      size: 64
      source: random
    - name: b
      size: 64
      source:
        type: file
        path: /etc/hostname
        strategy: from-start
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(
            cfg.collection.vectors[0].source,
            VectorSource::Random
        ));
        match &cfg.collection.vectors[1].source {
            VectorSource::File { path, strategy } => {
                assert_eq!(path, "/etc/hostname");
                assert!(matches!(strategy, FileStrategy::FromStart));
            }
            _ => panic!("expected file source"),
        }
    }

    #[test]
    fn parses_payloads_and_quantization() {
        let yaml = r#"
collection:
  quantization:
    type: turbo-4bit
    always_ram: true
  vectors:
    - size: 256
  fields:
    - name: color
      type: keyword
      source:
        type: random
        cardinality: 100
        values_per_point: 3
    - name: blob
      type: text
      index: false
      source: random
    - name: loc
      type: geo
      source:
        type: random-clusters
        clusters: 10
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.collection.quantization.as_ref().unwrap().kind,
            QuantKind::Turbo4bit
        );
        assert_eq!(cfg.collection.fields.len(), 3);
        assert_eq!(
            cfg.collection.fields[0]
                .source
                .as_ref()
                .unwrap()
                .cardinality,
            Some(100)
        );
        assert!(!cfg.collection.fields[1].index);
        assert_eq!(
            cfg.collection.fields[2].source.as_ref().unwrap().kind,
            PayloadSourceKind::RandomClusters
        );
    }

    #[test]
    fn whole_payload_source_allows_index_only_fields() {
        let yaml = r#"
collection:
  vectors:
    - size: 8
  payload:
    source:
      type: dataset
      dataset:
        name: laion-small-clip
        format: tar
        path: laion-small-clip/laion-small-clip
        link: https://example.com/laion-small-clip.tgz
  fields:
    - name: similarity
      type: float
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.collection.payload.source.is_some());
        assert!(cfg.collection.fields[0].source.is_none());
    }

    #[test]
    fn payload_source_must_be_dataset_without_field() {
        let with_field = r#"
collection:
  vectors:
    - size: 8
  payload:
    source:
      type: dataset
      field: similarity
      dataset:
        name: d
        format: tar
        path: d/d
        link: https://example.com/d.tgz
"#;
        let cfg: UploadConfig = serde_yaml::from_str(with_field).unwrap();
        assert!(cfg.validate().is_err());

        let not_dataset = r#"
collection:
  vectors:
    - size: 8
  payload:
    source:
      type: random
"#;
        let cfg: UploadConfig = serde_yaml::from_str(not_dataset).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn parses_dataset_vector_source() {
        let yaml = r#"
collection:
  vectors:
    - size: 25
      source:
        type: dataset
        name: glove-25-angular
        format: h5
        path: glove-25-angular/glove-25-angular.hdf5
        link: http://ann-benchmarks.com/glove-25-angular.hdf5
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().unwrap();
        assert!(matches!(
            cfg.collection.vectors[0].source,
            VectorSource::Dataset { .. }
        ));
    }

    #[test]
    fn parses_laion_small_clip_example() {
        let cfg = crate::config::examples::lookup("upload-laion-small-clip")
            .map(|e| super::parse(e.yaml, e.name).unwrap())
            .unwrap();
        // Dense vectors come from the tar dataset.
        assert!(matches!(
            cfg.collection.vectors[0].source,
            VectorSource::Dataset { .. }
        ));
        // Whole payload comes from the `payload.source` dataset.
        let src = cfg.collection.payload.source.as_ref().unwrap();
        assert_eq!(src.kind, PayloadSourceKind::Dataset);
        assert!(src.dataset.is_some());
        // The `similarity` field is an index-only declaration (no per-field source).
        let p = &cfg.collection.fields[0];
        assert_eq!(p.name, "similarity");
        assert_eq!(p.kind, PayloadType::Float);
        assert!(p.index);
        assert!(p.source.is_none());
    }

    /// The shipped example advertises every knob; keep it loadable so a config
    /// copied out of it actually works.
    #[test]
    fn parses_upload_config_example() {
        let cfg = crate::config::examples::lookup("upload-config")
            .map(|e| super::parse(e.yaml, e.name).unwrap())
            .unwrap();
        assert_eq!(
            cfg.collection.payload.memory,
            Some(crate::config::MemoryKind::Cached)
        );
        assert_eq!(
            cfg.collection.hnsw.as_ref().unwrap().memory,
            Some(crate::config::MemoryKind::Cold)
        );
        let color = cfg
            .collection
            .fields
            .iter()
            .find(|f| f.name == "color")
            .unwrap();
        assert!(color.prefix);
    }

    #[test]
    fn rejects_unknown_field() {
        let yaml = r#"
collection:
  vectors:
    - size: 128
      bogus: 1
"#;
        assert!(serde_yaml::from_str::<UploadConfig>(yaml).is_err());
    }

    fn config_with_vector_path(path: &str) -> UploadConfig {
        let yaml = format!(
            r#"
collection:
  vectors:
    - size: 128
      source:
        type: file
        path: {path}
"#
        );
        serde_yaml::from_str(&yaml).unwrap()
    }

    #[test]
    fn accepts_http_vector_source_path() {
        // Remote URLs are fetched lazily, so validation must not stat them.
        config_with_vector_path("https://example.com/vectors.fbin")
            .validate()
            .unwrap();
        config_with_vector_path("http://example.com/vectors.fbin")
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_s3_vector_source_path() {
        let err = config_with_vector_path("s3://bucket/vectors.fbin")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheme"), "{err}");
        assert!(err.contains("s3"), "{err}");
    }

    #[test]
    fn rejects_missing_local_vector_source_path() {
        let err = config_with_vector_path("/nonexistent/vectors.fbin")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn rejects_multiple_unnamed_vectors() {
        let yaml = r#"
collection:
  vectors:
    - size: 64
    - size: 64
"#;
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_empty_collection() {
        let yaml = "collection: {}\n";
        let cfg: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_err());
    }
}
