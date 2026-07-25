//! YAML-config-driven collection creation.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    BinaryQuantizationBuilder, BinaryQuantizationEncoding, BoolIndexParamsBuilder,
    CompressionRatio, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    CreateShardKeyBuilder, CreateShardKeyRequestBuilder, Datatype, DatetimeIndexParamsBuilder,
    Distance, FieldType, FloatIndexParamsBuilder, GeoIndexParamsBuilder, HnswConfigDiffBuilder,
    IntegerIndexParamsBuilder, KeywordIndexParamsBuilder, Memory, Modifier, MultiVectorComparator,
    MultiVectorConfigBuilder, OptimizersConfigDiffBuilder, PayloadStorageParamsBuilder,
    ProductQuantizationBuilder, QuantizationType, ScalarQuantizationBuilder, ShardingMethod,
    SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParamsBuilder,
    TextIndexParamsBuilder, TokenizerType, TurboQuantBitSize, TurboQuantizationBuilder,
    UuidIndexParamsBuilder, VectorParams, VectorParamsBuilder, VectorParamsMap, VectorsConfig,
    quantization_config,
};
use tokio::time::sleep;

use crate::args::Args;
use crate::client::random_client;
use crate::config::{
    ComparatorKind, DatatypeKind, DistanceKind, MemoryKind, ModifierKind, PayloadType, QuantKind,
    TokenizerKind, UploadConfig, VectorConfig,
};

fn distance_to_grpc(distance: DistanceKind) -> Distance {
    match distance {
        DistanceKind::Cosine => Distance::Cosine,
        DistanceKind::Dot => Distance::Dot,
        DistanceKind::Euclid => Distance::Euclid,
        DistanceKind::Manhattan => Distance::Manhattan,
    }
}

fn datatype_to_i32(datatype: DatatypeKind) -> i32 {
    match datatype {
        DatatypeKind::Float32 => Datatype::Float32.into(),
        DatatypeKind::Float16 => Datatype::Float16.into(),
        DatatypeKind::Uint8 => Datatype::Uint8.into(),
        DatatypeKind::Turbo4 => Datatype::Turbo4.into(),
    }
}

/// Sparse value modifier (`modifier:`). `None` ⇒ leave it unset on the request.
fn modifier_to_grpc(modifier: ModifierKind) -> Option<Modifier> {
    match modifier {
        ModifierKind::None => None,
        ModifierKind::Idf => Some(Modifier::Idf),
    }
}

/// Memory placement (`memory:`), as the gRPC enum value.
pub fn memory_to_i32(memory: MemoryKind) -> i32 {
    match memory {
        MemoryKind::Cold => Memory::Cold.into(),
        MemoryKind::Cached => Memory::Cached.into(),
        MemoryKind::Pinned => Memory::Pinned.into(),
    }
}

pub fn build_quantization_config(
    kind: QuantKind,
    always_ram: bool,
    memory: Option<MemoryKind>,
) -> Option<quantization_config::Quantization> {
    // `memory` supersedes `always_ram`; both are sent so older servers still
    // honour the boolean.
    let config: quantization_config::Quantization = match kind {
        QuantKind::None => return None,
        QuantKind::Scalar => {
            let mut builder = ScalarQuantizationBuilder::default()
                .r#type(QuantizationType::Int8.into())
                .quantile(0.99)
                .always_ram(always_ram);
            if let Some(memory) = memory {
                builder = builder.memory(memory_to_i32(memory));
            }
            builder.into()
        }
        QuantKind::Binary | QuantKind::Binary2bit | QuantKind::Binary15bit => {
            let mut builder = BinaryQuantizationBuilder::new(always_ram);
            builder = match kind {
                QuantKind::Binary2bit => builder.encoding(BinaryQuantizationEncoding::TwoBits),
                QuantKind::Binary15bit => {
                    builder.encoding(BinaryQuantizationEncoding::OneAndHalfBits)
                }
                _ => builder,
            };
            if let Some(memory) = memory {
                builder = builder.memory(memory_to_i32(memory));
            }
            builder.into()
        }
        QuantKind::Turbo1bit
        | QuantKind::Turbo15bit
        | QuantKind::Turbo2bit
        | QuantKind::Turbo4bit => {
            let bits = match kind {
                QuantKind::Turbo1bit => TurboQuantBitSize::Bits1,
                QuantKind::Turbo15bit => TurboQuantBitSize::Bits15,
                QuantKind::Turbo2bit => TurboQuantBitSize::Bits2,
                _ => TurboQuantBitSize::Bits4,
            };
            let mut builder = TurboQuantizationBuilder::new()
                .bits(bits)
                .always_ram(always_ram);
            if let Some(memory) = memory {
                builder = builder.memory(memory_to_i32(memory));
            }
            builder.into()
        }
        QuantKind::ProductX4
        | QuantKind::ProductX8
        | QuantKind::ProductX16
        | QuantKind::ProductX32
        | QuantKind::ProductX64 => {
            let compression = match kind {
                QuantKind::ProductX4 => CompressionRatio::X4,
                QuantKind::ProductX8 => CompressionRatio::X8,
                QuantKind::ProductX16 => CompressionRatio::X16,
                QuantKind::ProductX32 => CompressionRatio::X32,
                _ => CompressionRatio::X64,
            };
            let mut builder =
                ProductQuantizationBuilder::new(compression.into()).always_ram(always_ram);
            if let Some(memory) = memory {
                builder = builder.memory(memory_to_i32(memory));
            }
            builder.into()
        }
    };
    Some(config)
}

fn build_vector_params(vc: &VectorConfig) -> VectorParams {
    let mut builder = VectorParamsBuilder::new(vc.size, distance_to_grpc(vc.distance))
        .datatype(datatype_to_i32(vc.datatype));

    if let Some(on_disk) = vc.on_disk {
        builder = builder.on_disk(on_disk);
    }
    if let Some(memory) = vc.memory {
        builder = builder.memory(memory_to_i32(memory));
    }
    if let Some(mv) = &vc.multivector {
        let comparator = match mv.comparator {
            ComparatorKind::MaxSim => MultiVectorComparator::MaxSim,
        };
        builder = builder.multivector_config(MultiVectorConfigBuilder::new(comparator));
    }
    if let Some(quantization) = vc
        .quantization
        .as_ref()
        .and_then(|q| build_quantization_config(q.kind, q.always_ram, q.memory))
    {
        builder = builder.quantization_config(quantization);
    }

    builder.build()
}

pub async fn recreate_collection_from_config(
    config: &UploadConfig,
    args: &Args,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let client = random_client(args)?;
    let collection = &config.collection;

    if args.create_if_missing && client.collection_exists(&collection.name).await? {
        println!("Collection {} already exists", collection.name);
        return Ok(());
    }

    match client.delete_collection(&collection.name).await {
        Ok(_) => println!("Deleted collection: {}", collection.name),
        Err(e) => println!("Failed to delete collection: {e:?}"),
    }

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }
    sleep(Duration::from_secs(1)).await;
    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    // Dense vectors config: a single unnamed vector uses `Params`, otherwise a
    // named `ParamsMap` (the unnamed default vector, if any, keyed by "").
    let mut params_map: HashMap<String, VectorParams> = HashMap::new();
    let mut default_param: Option<VectorParams> = None;
    for vc in &collection.vectors {
        let params = build_vector_params(vc);
        match &vc.name {
            Some(name) => {
                params_map.insert(name.clone(), params);
            }
            None => default_param = Some(params),
        }
    }
    let vectors_config: Option<VectorsConfig> = if params_map.is_empty() {
        default_param.map(|p| Config::Params(p).into())
    } else {
        if let Some(p) = default_param {
            params_map.insert("".to_string(), p);
        }
        Some(Config::ParamsMap(VectorParamsMap { map: params_map }).into())
    };

    // Sparse vectors config.
    let sparse_vectors_config = if collection.sparse_vectors.is_empty() {
        None
    } else {
        let params: HashMap<_, _> = collection
            .sparse_vectors
            .iter()
            .map(|sc| {
                let mut index_builder = SparseIndexConfigBuilder::default()
                    .on_disk(sc.on_disk)
                    .datatype(datatype_to_i32(sc.datatype));
                if let Some(memory) = sc.memory {
                    index_builder = index_builder.memory(memory_to_i32(memory));
                }
                let mut params = SparseVectorParamsBuilder::default().index(index_builder);
                if let Some(modifier) = modifier_to_grpc(sc.modifier) {
                    params = params.modifier(modifier);
                }
                (sc.name.clone(), params.build())
            })
            .collect();
        Some(SparseVectorConfig::from(params))
    };

    let mut create_collection_builder = CreateCollectionBuilder::new(collection.name.clone())
        .on_disk_payload(collection.on_disk_payload)
        .replication_factor(collection.replication_factor)
        .write_consistency_factor(collection.write_consistency_factor);

    if let Some(vectors_config) = vectors_config {
        create_collection_builder = create_collection_builder.vectors_config(vectors_config);
    }
    if let Some(sparse_vectors_config) = sparse_vectors_config {
        create_collection_builder =
            create_collection_builder.sparse_vectors_config(sparse_vectors_config);
    }
    if let Some(shard_number) = collection.shard_number {
        create_collection_builder = create_collection_builder.shard_number(shard_number);
    }
    if let Some(memory) = collection.payload.memory {
        create_collection_builder = create_collection_builder
            .payload(PayloadStorageParamsBuilder::default().memory(memory_to_i32(memory)));
    }

    if let Some(hnsw) = &collection.hnsw {
        let mut hnsw_config = HnswConfigDiffBuilder::default()
            .on_disk(hnsw.on_disk)
            .inline_storage(hnsw.inline_storage);
        if let Some(memory) = hnsw.memory {
            hnsw_config = hnsw_config.memory(memory_to_i32(memory));
        }
        if let Some(m) = hnsw.m {
            hnsw_config = hnsw_config.m(m);
        }
        if let Some(payload_m) = hnsw.payload_m {
            hnsw_config = hnsw_config.payload_m(payload_m);
        }
        if let Some(ef_construct) = hnsw.ef_construct {
            hnsw_config = hnsw_config.ef_construct(ef_construct);
        }
        if let Some(fst) = hnsw.full_scan_threshold {
            hnsw_config = hnsw_config.full_scan_threshold(fst);
        }
        create_collection_builder = create_collection_builder.hnsw_config(hnsw_config);
    }

    if let Some(opt) = &collection.optimizers {
        let mut optimizers_config = OptimizersConfigDiffBuilder::default();
        if let Some(n) = opt.default_segment_number {
            optimizers_config = optimizers_config.default_segment_number(n);
        }
        if let Some(t) = opt.memmap_threshold {
            optimizers_config = optimizers_config.memmap_threshold(t);
        }
        if let Some(t) = opt.indexing_threshold {
            optimizers_config = optimizers_config.indexing_threshold(t);
        }
        if let Some(s) = opt.max_segment_size {
            optimizers_config = optimizers_config.max_segment_size(s);
        }
        if opt.prevent_unoptimized {
            optimizers_config = optimizers_config.prevent_unoptimized(true);
        }
        create_collection_builder = create_collection_builder.optimizers_config(optimizers_config);
    }

    if let Some(quant) = &collection.quantization
        && let Some(quant_config) =
            build_quantization_config(quant.kind, quant.always_ram, quant.memory)
    {
        create_collection_builder = create_collection_builder.quantization_config(quant_config);
    }

    if collection.sharding.is_some() {
        create_collection_builder =
            create_collection_builder.sharding_method(ShardingMethod::Custom.into());
    }

    client.create_collection(create_collection_builder).await?;
    println!("Created collection: {}", collection.name);

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }
    sleep(Duration::from_secs(1)).await;

    if !args.skip_field_indices {
        create_field_indices_from_config(config, &client).await?;
    }

    if let Some(sharding) = &collection.sharding {
        let mut builder = CreateShardKeyBuilder::default()
            .shard_key(Key::Keyword(sharding.key.clone()))
            .replication_factor(collection.replication_factor);
        if let Some(shards) = collection.shard_number {
            builder = builder.shards_number(shards);
        }
        client
            .create_shard_key(
                CreateShardKeyRequestBuilder::new(collection.name.clone()).request(builder),
            )
            .await?;
    }

    Ok(())
}

async fn create_field_indices_from_config(
    config: &UploadConfig,
    client: &qdrant_client::Qdrant,
) -> Result<()> {
    for pc in &config.collection.fields {
        if !pc.index {
            continue;
        }

        let name = config.collection.name.clone();
        let field = pc.name.clone();

        // `memory` supersedes `on_disk` on servers that understand it; both are
        // sent so the config keeps working against older ones.
        let memory = pc.memory.map(memory_to_i32);

        let builder = match pc.kind {
            PayloadType::Keyword => {
                let mut params = KeywordIndexParamsBuilder::default()
                    .on_disk(pc.on_disk)
                    .is_tenant(pc.is_tenant)
                    .prefix(pc.prefix);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Keyword)
                    .field_index_params(params)
            }
            PayloadType::Integer => {
                let mut params = IntegerIndexParamsBuilder::new(true, pc.range_index)
                    .on_disk(pc.on_disk)
                    .is_principal(pc.is_principal);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Integer)
                    .field_index_params(params)
            }
            PayloadType::Float => {
                let mut params = FloatIndexParamsBuilder::default()
                    .on_disk(pc.on_disk)
                    .is_principal(pc.is_principal);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Float)
                    .field_index_params(params)
            }
            PayloadType::Bool => {
                let mut params = BoolIndexParamsBuilder::default().on_disk(pc.on_disk);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Bool)
                    .field_index_params(params)
            }
            PayloadType::Uuid => {
                let mut params = UuidIndexParamsBuilder::default()
                    .is_tenant(pc.is_tenant)
                    .on_disk(pc.on_disk);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Uuid)
                    .field_index_params(params)
            }
            PayloadType::Geo => {
                let mut params = GeoIndexParamsBuilder::new().on_disk(pc.on_disk);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Geo)
                    .field_index_params(params)
            }
            PayloadType::Datetime => {
                let mut params = DatetimeIndexParamsBuilder::default()
                    .on_disk(pc.on_disk)
                    .is_principal(pc.is_principal);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Datetime)
                    .field_index_params(params)
            }
            PayloadType::Text => {
                let tokenizer = match pc.tokenizer.unwrap_or(TokenizerKind::Word) {
                    TokenizerKind::Word => TokenizerType::Word,
                    TokenizerKind::Whitespace => TokenizerType::Whitespace,
                    TokenizerKind::Prefix => TokenizerType::Prefix,
                    TokenizerKind::Multilingual => TokenizerType::Multilingual,
                };
                let mut params = TextIndexParamsBuilder::new(tokenizer).on_disk(pc.on_disk);
                if let Some(memory) = memory {
                    params = params.memory(memory);
                }
                CreateFieldIndexCollectionBuilder::new(name, field, FieldType::Text)
                    .field_index_params(params)
            }
        };

        client.create_field_index(builder.wait(true)).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_config(yaml: &str) -> VectorConfig {
        let config: UploadConfig = serde_yaml::from_str(yaml).unwrap();
        config.collection.vectors.into_iter().next().unwrap()
    }

    #[test]
    fn distance_mapping() {
        assert_eq!(distance_to_grpc(DistanceKind::Cosine), Distance::Cosine);
        assert_eq!(distance_to_grpc(DistanceKind::Dot), Distance::Dot);
        assert_eq!(distance_to_grpc(DistanceKind::Euclid), Distance::Euclid);
        assert_eq!(
            distance_to_grpc(DistanceKind::Manhattan),
            Distance::Manhattan
        );
    }

    #[test]
    fn datatype_mapping() {
        assert_eq!(
            datatype_to_i32(DatatypeKind::Float32),
            Datatype::Float32 as i32
        );
        assert_eq!(
            datatype_to_i32(DatatypeKind::Float16),
            Datatype::Float16 as i32
        );
        assert_eq!(datatype_to_i32(DatatypeKind::Uint8), Datatype::Uint8 as i32);
        assert_eq!(
            datatype_to_i32(DatatypeKind::Turbo4),
            Datatype::Turbo4 as i32
        );
    }

    #[test]
    fn memory_mapping() {
        assert_eq!(memory_to_i32(MemoryKind::Cold), Memory::Cold as i32);
        assert_eq!(memory_to_i32(MemoryKind::Cached), Memory::Cached as i32);
        assert_eq!(memory_to_i32(MemoryKind::Pinned), Memory::Pinned as i32);
    }

    #[test]
    fn quantization_none_is_none() {
        assert!(build_quantization_config(QuantKind::None, true, None).is_none());
    }

    #[test]
    fn quantization_variants_build() {
        assert!(build_quantization_config(QuantKind::Scalar, true, None).is_some());
        assert!(build_quantization_config(QuantKind::Binary, false, None).is_some());
        assert!(build_quantization_config(QuantKind::ProductX8, false, None).is_some());
        assert!(build_quantization_config(QuantKind::Turbo4bit, true, None).is_some());
    }

    /// Every quantization kind must carry `memory` through to the gRPC message.
    #[test]
    fn quantization_memory_is_passed_through() {
        use qdrant_client::qdrant::quantization_config::Quantization;

        let expected = Memory::Pinned as i32;
        for kind in [
            QuantKind::Scalar,
            QuantKind::Binary,
            QuantKind::Binary2bit,
            QuantKind::Binary15bit,
            QuantKind::Turbo1bit,
            QuantKind::Turbo15bit,
            QuantKind::Turbo2bit,
            QuantKind::Turbo4bit,
            QuantKind::ProductX4,
            QuantKind::ProductX8,
            QuantKind::ProductX16,
            QuantKind::ProductX32,
            QuantKind::ProductX64,
        ] {
            let config = build_quantization_config(kind, false, Some(MemoryKind::Pinned)).unwrap();
            let memory = match config {
                Quantization::Scalar(q) => q.memory,
                Quantization::Binary(q) => q.memory,
                Quantization::Turboquant(q) => q.memory,
                Quantization::Product(q) => q.memory,
            };
            assert_eq!(memory, Some(expected), "{kind:?} dropped `memory`");
        }
    }

    #[test]
    fn vector_params_carry_memory_and_on_disk() {
        let vc = vector_config(
            "collection:\n  vectors:\n    - size: 8\n      on_disk: true\n      memory: cached\n",
        );
        let params = build_vector_params(&vc);
        assert_eq!(params.memory, Some(Memory::Cached as i32));
        #[expect(deprecated)]
        {
            assert_eq!(params.on_disk, Some(true));
        }
    }

    #[test]
    fn vector_params_from_config() {
        let vc = vector_config(
            "collection:\n  vectors:\n    - size: 256\n      distance: dot\n      datatype: uint8\n      quantization: { type: scalar }\n      multivector: { count: 4 }\n",
        );
        let params = build_vector_params(&vc);
        assert_eq!(params.size, 256);
        assert_eq!(params.distance, Distance::Dot as i32);
        assert_eq!(params.datatype, Some(Datatype::Uint8 as i32));
        assert!(params.multivector_config.is_some());
        assert!(params.quantization_config.is_some());
    }
}
