//! Human-readable schema reference for the upload-config file
//! (`bfb upload --file config.yaml`). Printed by the `bfb schema` subcommand.
//!
//! The prose/annotations are hand-written, but completeness is enforced: the
//! `reference_covers_every_field` test builds a fully-populated
//! [`UploadConfig`](crate::config::UploadConfig) from explicit struct literals
//! (so adding/renaming a field is a compile error there), serializes it, and
//! asserts every real field name appears in the text below — so the reference
//! cannot silently fall out of sync with the structs in the `config` module.

/// Print the annotated YAML schema reference to stdout.
pub fn print_schema() {
    print!("{SCHEMA_REFERENCE}");
}

const SCHEMA_REFERENCE: &str = r#"# bfb upload-config file schema (`bfb upload --file <config.yaml>`)
#
# The file describes only the *shape* of the data: collection parameters plus
# how each field's values are generated. The *how* of uploading (number of
# points `-n`, batch size `-b`, parallelism `-p`, threads `-t`, `--uri`, …)
# stays on the CLI.
#
# Legend:  <type>  default=<value>  [allowed | values]   (Option ⇒ optional)
# Unknown fields are rejected. At least one dense or sparse vector is required.
#
# `memory:` (Qdrant 1.19+) is accepted wherever an `on_disk` / `always_ram`
# boolean is, and supersedes it when the server understands it:
#   cold    data is not pre-loaded from disk to RAM; cached with usage
#   cached  pre-loaded into disk-cache RAM on start, may be evicted under pressure
#   pinned  loaded in RAM and never evicted (not supported for dense vectors
#           or the payload storage)
# Both are sent, so configs stay usable against older servers.

collection:
  name: benchmark                # string         default="benchmark"  collection name
  id: integer                    # enum           default=integer      [integer | uuid] point-id type
  on_disk_payload: true          # bool           default=true         store payload on disk
  shard_number: null             # uint32         optional             number of shards
  replication_factor: 1          # uint32         default=1
  write_consistency_factor: 1    # uint32         default=1

  # Custom sharding (optional). Only the `custom` method is supported.
  sharding:
    method: custom               # string         default="custom"     [custom]
    key: my_key                  # string         required             payload field used as shard key

  # HNSW index parameters (optional). Omitted fields fall back to server defaults.
  hnsw:
    m: null                      # uint64         optional             edges per node
    payload_m: null              # uint64         optional             edges per node for payload subgraph
    ef_construct: null           # uint64         optional             beam size at construction
    full_scan_threshold: null    # uint64         optional
    on_disk: false               # bool           default=false        store HNSW graph on disk
    inline_storage: false        # bool           default=false
    memory: null                 # enum           optional             [cold | cached | pinned] supersedes `on_disk`

  # Optimizer parameters (optional).
  optimizers:
    default_segment_number: null # uint64         optional
    indexing_threshold: null     # uint64         optional
    memmap_threshold: null       # uint64         optional
    max_segment_size: null       # uint64         optional
    deleted_threshold: null      # double         optional             fraction of a segment that must be
                                 #   deleted before the vacuum optimizer rebuilds it (server default 0.2)
    vacuum_min_vector_number: null # uint64       optional             smallest segment, in vectors, the
                                 #   vacuum optimizer will consider (server default 1000)
    prevent_unoptimized: false   # bool           default=false        wait for a fully optimized collection

  # Collection-wide quantization (optional). Also settable per dense vector.
  quantization:
    type: scalar                 # enum           required             [none | scalar | binary |
                                 #   binary-2bit | binary-1.5bit | turbo-1bit | turbo-1.5bit |
                                 #   turbo-2bit | turbo-4bit | product-x4 | product-x8 |
                                 #   product-x16 | product-x32 | product-x64]
    always_ram: false            # bool           default=false        keep quantized vectors in RAM
    memory: null                 # enum           optional             [cold | cached | pinned] supersedes `always_ram`

  # Dense vectors. At most one may omit `name` (the unnamed default vector);
  # otherwise every entry must have a unique `name`.
  vectors:
    - name: image                # string         optional             omit for the unnamed default vector
      size: 512                  # uint64         required             dimensionality
      distance: cosine           # enum           default=cosine       [cosine | dot | euclid | manhattan]
      datatype: float32          # enum           default=float32      [float32 | float16 | uint8 | turbo4]
      on_disk: null              # bool           optional             store vectors on disk
      memory: null               # enum           optional             [cold | cached] supersedes `on_disk`
      quantization: null         # map            optional             same shape as `collection.quantization`
      # Multivectors (optional): generate several sub-vectors per point.
      multivector:
        comparator: max_sim      # enum           default=max_sim      [max_sim]
        count: 4                 # uint           required             sub-vectors per point
      # Value source. Shorthand string `random`, or a map:
      source: random             # default=random
      # source:
      #   type: file             # enum           [random | file | dataset]
      #   path: ./vectors.fbin   # string         required for file (local path or http(s):// URL, cached on download)
      #   strategy: random-sample # enum          default=random-sample  [random-sample | from-start]
      # source:
      #   type: dataset          # inline dataset definition (vector-db-benchmark format)
      #   name: glove-25-angular
      #   format: h5             # dataset format (`type` alias accepted in nested `dataset:` maps):
      #                          #   h5      ann-benchmarks bundle (train/test/neighbors)
      #                          #   tar     .tgz of vectors.npy + payloads.jsonl + tests.jsonl
      #                          #   sparse  CSR matrices
      #                          #   npy     one 2-D float .npy — dense vectors only
      #                          #   parquet one parquet file — payload rows only
      #   path: glove-25-angular/glove-25-angular.hdf5
      #   link: http://ann-benchmarks.com/glove-25-angular.hdf5
      #   vector_size: 25
      #   distance: cosine
      # A sharded dataset (`npy` / `parquet` only) replaces `path`/`link` with a
      # `parts` block; the files are read as one row space and `{i}` is
      # substituted with each part's number. Row counts per part are measured,
      # not configured — one ranged request per part, cached thereafter.
      # source:
      #   type: dataset
      #   name: laion-400m-img-emb
      #   format: npy
      #   parts:
      #     count: 410           # uint     required   number of parts
      #     start: 0             # uint     default=0  index of the first part
      #     path: laion/img_emb_{i}.npy     # string   required
      #     link: https://host/img_emb_{i}.npy  # string  optional
      #   cache: keep          # enum  default=keep  [keep | evict] (sharded only)
      #                        #   evict deletes each downloaded part once the reader
      #                        #   moves past it, and prefetches the next one, so a
      #                        #   corpus larger than the disk can still be streamed.
      #                        #   Only parts bfb downloaded are ever deleted.

  # Sparse vectors (optional). Names must be unique across all vectors.
  sparse_vectors:
    - name: bm25                 # string         required
      datatype: float32          # enum           default=float32      [float32 | float16 | uint8]
      on_disk: false             # bool           default=false
      memory: null               # enum           optional             [cold | cached | pinned] supersedes `on_disk`
      modifier: none             # enum           default=none         [none | idf] `idf` enables BM25-style
                                 #   scoring, and is required by search `idf_corpus`
      # Value source. Shorthand string `random`, or a map:
      source:
        type: random             # enum           default=random       [random | dataset]
        vocab_size: 1000            # uint           default=1000         vocabulary size (max sparse index)
        length: 100                 # uint           default=100          number of non-zero entries
        distribution: uniform    # enum           default=uniform      [uniform | zipf]
      # source:
      #   type: dataset
      #   dataset:
      #     name: msmarco-sparse-100K
      #     format: sparse
      #     path: msmarco-sparse-100K/data.csr
      #     link: https://example.com/msmarco-sparse-100K.tgz

  # Payload-wide settings (optional). `payload.source` is a whole-payload source:
  # when set to `type: dataset`, each point's entire payload object is loaded from
  # the dataset's `payloads.jsonl`. The `fields` below then only declare which
  # fields to index (and may omit their own `source`); fields present in the
  # object but not listed are uploaded but left unindexed.
  payload:
    memory: null                 # enum           optional             [cold | cached] supersedes `on_disk_payload`
    source: null                 # optional            whole-payload dataset source, e.g.:
    #   type: dataset
    #   dataset:
    #     name: laion-small-clip
    #     format: tar
    #     path: laion-small-clip/laion-small-clip
    #     link: https://example.com/laion-small-clip.tgz
    # `format: parquet` reads payload rows from a parquet file, and accepts
    # three extra keys (ignored by every other format):
    #   columns: [url, similarity]   # list  optional  columns to keep (default: all)
    #   exclude: [exif]              # list  default=[]  columns to drop (applied after `columns`)
    #   fill_null: 0                 # any   optional  value substituted for nulls and for
    #                                #   NaN/±inf floats, which have no JSON form. Omitted by
    #                                #   default, leaving the payload field absent.

  # Payload field declarations (optional). Names must be unique. Each entry
  # generates a value and/or declares a field index.
  fields:
    - name: color                # string         required
      type: keyword              # enum           required             [keyword | integer | float |
                                 #   bool | uuid | geo | text | datetime]
      index: true                # bool           default=true         build a field index (false ⇒ filler)
      on_disk: false             # bool           default=false        store the index on disk
      memory: null               # enum           optional             [cold | cached | pinned] supersedes `on_disk`
      is_tenant: false           # bool           default=false        tenant-isolating index
      is_principal: false        # bool           default=false        principal (primary) index
      range_index: true          # bool           default=true         integer payloads: also build a range index
      prefix: false              # bool           default=false        keyword payloads: enable prefix matching
                                 #   (required for search `match_prefix` filters)
      tokenizer: null            # enum           optional (text)      [word | whitespace | prefix | multilingual]
      # Value source (optional when `payload.source` is set — then the entry is
      # index-only). Shorthand string `random` / `random-clusters` / `now`, or a
      # map. Which keys apply depends on the payload `type`; others ignored.
      source:
        type: random             # enum           default=random       [random | random-clusters | now | dataset]
        cardinality: null        # uint           optional (keyword)   number of distinct values
        length_multiplier: null  # uint           optional (keyword)   value-length multiplier
        values_per_point: null   # uint           optional (keyword/integer)  multivalue count per point
        min: null                # float          optional (integer/float/datetime range)
        max: null                # float          optional (integer/float/datetime range)
        true_ratio: null         # float          optional (bool)      fraction of `true` values
        clusters: null           # uint           optional (geo)       number of geo clusters (> 0)
        vocab_size: null         # uint           optional (text)
        min_length: null         # uint           optional (text)
        max_length: null         # uint           optional (text)
        distribution: uniform    # enum           default=uniform      [uniform | zipf]
"#;

#[cfg(test)]
mod tests {
    use crate::config::collection::*;
    use crate::config::payload::*;
    use crate::config::vector::*;
    use crate::config::*;

    /// A fully-populated config touching *every* field of *every* config
    /// struct. Built from explicit struct literals on purpose: adding,
    /// removing, or renaming a field anywhere in the `config` module turns this into a
    /// compile error, forcing a conscious update of the schema reference.
    fn reference_example() -> UploadConfig {
        UploadConfig {
            collection: CollectionConfig {
                name: "benchmark".to_string(),
                id: IdType::Uuid,
                on_disk_payload: true,
                shard_number: Some(1),
                replication_factor: 1,
                write_consistency_factor: 1,
                sharding: Some(ShardingConfig {
                    method: "custom".to_string(),
                    key: "my_key".to_string(),
                }),
                hnsw: Some(HnswConfig {
                    m: Some(16),
                    payload_m: Some(16),
                    ef_construct: Some(100),
                    full_scan_threshold: Some(10000),
                    on_disk: false,
                    inline_storage: false,
                    memory: Some(MemoryKind::Cached),
                }),
                optimizers: Some(OptimizersConfig {
                    default_segment_number: Some(2),
                    indexing_threshold: Some(20000),
                    memmap_threshold: Some(20000),
                    max_segment_size: Some(200000),
                    deleted_threshold: Some(0.2),
                    vacuum_min_vector_number: Some(1000),
                    prevent_unoptimized: false,
                }),
                quantization: Some(QuantizationConfig {
                    kind: QuantKind::Scalar,
                    always_ram: false,
                    memory: Some(MemoryKind::Pinned),
                }),
                vectors: vec![VectorConfig {
                    name: Some("image".to_string()),
                    size: 512,
                    distance: DistanceKind::Cosine,
                    datatype: DatatypeKind::Float32,
                    on_disk: Some(false),
                    memory: Some(MemoryKind::Cached),
                    multivector: Some(MultivectorConfig {
                        comparator: ComparatorKind::MaxSim,
                        count: 4,
                    }),
                    quantization: Some(QuantizationConfig {
                        kind: QuantKind::Scalar,
                        always_ram: false,
                        memory: Some(MemoryKind::Pinned),
                    }),
                    // File source so `path`/`strategy` are exercised too;
                    // a remote path keeps `validate()` from checking existence.
                    source: VectorSource::File {
                        path: "https://example.com/vectors.fbin".to_string(),
                        strategy: FileStrategy::RandomSample,
                    },
                }],
                sparse_vectors: vec![SparseVectorConfig {
                    name: "bm25".to_string(),
                    datatype: DatatypeKind::Float32,
                    on_disk: false,
                    memory: Some(MemoryKind::Cached),
                    modifier: ModifierKind::Idf,
                    source: SparseSource {
                        kind: SparseKind::Random,
                        vocab_size: 1000,
                        length: 100,
                        distribution: DistributionKind::Uniform,
                        dataset: None,
                    },
                }],
                payload: PayloadSection {
                    source: None,
                    memory: Some(MemoryKind::Cached),
                },
                fields: vec![PayloadConfig {
                    name: "color".to_string(),
                    kind: PayloadType::Keyword,
                    index: true,
                    on_disk: false,
                    memory: Some(MemoryKind::Cached),
                    is_tenant: false,
                    is_principal: false,
                    range_index: true,
                    prefix: true,
                    tokenizer: Some(TokenizerKind::Word),
                    source: Some(PayloadSource {
                        kind: PayloadSourceKind::Random,
                        dataset: None,
                        field: None,
                        cardinality: Some(100),
                        length_multiplier: Some(1),
                        values_per_point: Some(1),
                        min: Some(0.0),
                        max: Some(1.0),
                        true_ratio: Some(0.5),
                        clusters: Some(10),
                        vocab_size: Some(1000),
                        min_length: Some(1),
                        max_length: Some(10),
                        distribution: DistributionKind::Uniform,
                    }),
                }],
            },
        }
    }

    /// Recursively collect every mapping key from a YAML value.
    fn collect_keys(value: &serde_yaml::Value, out: &mut std::collections::HashSet<String>) {
        match value {
            serde_yaml::Value::Mapping(map) => {
                for (k, v) in map {
                    if let serde_yaml::Value::String(s) = k {
                        out.insert(s.clone());
                    }
                    collect_keys(v, out);
                }
            }
            serde_yaml::Value::Sequence(seq) => {
                for v in seq {
                    collect_keys(v, out);
                }
            }
            _ => {}
        }
    }

    /// The hand-written reference must mention every real config field, so it
    /// can't drift out of sync with the structs. `reference_example` (compile-
    /// checked) is the source of field names.
    #[test]
    fn reference_covers_every_field() {
        let example = reference_example();
        // Sanity: the example itself is a valid, parseable config.
        example.validate().unwrap();
        let yaml = serde_yaml::to_string(&example).unwrap();
        let value: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
        let _: UploadConfig = serde_yaml::from_str(&yaml).unwrap();

        let mut keys = std::collections::HashSet::new();
        collect_keys(&value, &mut keys);

        let missing: Vec<_> = keys
            .iter()
            .filter(|k| !super::SCHEMA_REFERENCE.contains(k.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "`bfb schema` reference is missing fields: {missing:?}\n\
             Add them to SCHEMA_REFERENCE in src/schema.rs."
        );
    }
}
