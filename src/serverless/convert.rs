//! Convert a BFB [`UploadConfig`] into a serverless [`CollectionConfig`].
//!
//! Serverless only accepts the tenant-facing shape (dense/sparse vectors +
//! payload indexes). Storage knobs from the upload YAML (HNSW, quantization,
//! on-disk placement, …) are ignored — the serverless manager decides those.

use anyhow::{Result, bail};
use qdrant_client::serverless::{
    BoolIndex, CollectionConfig, DenseVectorConfig, Distance, FloatIndex, GeoIndex, IntegerIndex,
    KeywordIndex, PayloadIndex, SparseVectorConfig, TextIndex, Tokenizer, UuidIndex,
};

use crate::config::payload::PayloadType;
use crate::config::vector::{DistanceKind, ModifierKind};
use crate::config::{TokenizerKind, UploadConfig};

/// Map an upload-shape YAML into the serverless create-collection config.
pub fn to_serverless_config(upload: &UploadConfig) -> Result<CollectionConfig> {
    let c = &upload.collection;
    if c.vectors.is_empty() && c.sparse_vectors.is_empty() {
        bail!("config must define at least one dense or sparse vector");
    }

    let mut config = CollectionConfig::new();

    for v in &c.vectors {
        let dense = DenseVectorConfig::new(v.size, map_distance(v.distance))
            .multivector(v.multivector.is_some());
        match &v.name {
            Some(name) => config = config.named_dense_vector(name.clone(), dense),
            None => config = config.dense_vector(dense),
        }
    }

    for s in &c.sparse_vectors {
        let sparse = SparseVectorConfig::new().use_idf(s.modifier == ModifierKind::Idf);
        config = config.named_sparse_vector(s.name.clone(), sparse);
    }

    for field in &c.fields {
        if !field.index {
            continue;
        }
        let index: PayloadIndex = match field.kind {
            PayloadType::Keyword => KeywordIndex.into(),
            PayloadType::Integer => IntegerIndex::new()
                .lookup(true)
                .range(field.range_index)
                .into(),
            PayloadType::Float => FloatIndex.into(),
            PayloadType::Bool => BoolIndex.into(),
            PayloadType::Uuid => UuidIndex.into(),
            PayloadType::Geo => GeoIndex.into(),
            PayloadType::Text => {
                let mut text = TextIndex::new();
                if let Some(tok) = field.tokenizer {
                    text = text.tokenizer(map_tokenizer(tok));
                }
                text.into()
            }
            PayloadType::Datetime => qdrant_client::serverless::DatetimeIndex.into(),
        };
        config = config.payload_index(field.name.clone(), index);
    }

    Ok(config)
}

fn map_distance(d: DistanceKind) -> Distance {
    match d {
        DistanceKind::Cosine => Distance::Cosine,
        DistanceKind::Dot => Distance::Dot,
        DistanceKind::Euclid => Distance::Euclid,
        DistanceKind::Manhattan => Distance::Manhattan,
    }
}

fn map_tokenizer(t: TokenizerKind) -> Tokenizer {
    match t {
        TokenizerKind::Word => Tokenizer::Word,
        TokenizerKind::Whitespace => Tokenizer::Whitespace,
        TokenizerKind::Prefix => Tokenizer::Prefix,
        TokenizerKind::Multilingual => Tokenizer::Multilingual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_basic_upload_config() {
        let yaml = r#"
collection:
  vectors:
    - size: 128
      distance: cosine
  fields:
    - name: color
      type: keyword
"#;
        let upload = crate::config::parse(yaml, "test").unwrap();
        let cfg = to_serverless_config(&upload).unwrap();
        assert_eq!(cfg.dense_vectors.len(), 1);
        assert!(cfg.dense_vectors.contains_key(""));
        assert_eq!(cfg.dense_vectors[""].size, 128);
        assert!(cfg.payload_indexes.contains_key("color"));
    }
}
