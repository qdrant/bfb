//! Flag-driven collection creation (legacy CLI path).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    BoolIndexParamsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    CreateShardKeyBuilder, CreateShardKeyRequestBuilder, Datatype, DatetimeIndexParamsBuilder,
    Distance, FieldType, FloatIndexParamsBuilder, GeoIndexParamsBuilder, HnswConfigDiffBuilder,
    IntegerIndexParamsBuilder, KeywordIndexParamsBuilder, Modifier, MultiVectorComparator,
    MultiVectorConfigBuilder, OptimizersConfigDiffBuilder, PayloadStorageParamsBuilder,
    ShardingMethod, SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParamsBuilder,
    TextIndexParamsBuilder, TokenizerType, UuidIndexParamsBuilder, VectorParamsBuilder,
    VectorParamsMap, VectorsConfig,
};
use tokio::time::sleep;

use super::from_config::{build_quantization_config, memory_to_i32};
use crate::args::Args;
use crate::client::random_client;
use crate::generators::random::{
    BOOL_PAYLOAD_KEY, FLOAT_PAYLOAD_KEY, GEO_PAYLOAD_KEY, INTEGERS_PAYLOAD_KEY,
    KEYWORD_PAYLOAD_KEY, TEXT_PAYLOAD_KEY, UUID_PAYLOAD_KEY, payload_prefixes,
};

pub async fn recreate_collection(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = random_client(args)?;

    if args.create_if_missing && client.collection_exists(&args.collection_name).await? {
        println!("Collection {} already exists", args.collection_name);
        return Ok(());
    }

    match client.delete_collection(&args.collection_name).await {
        Ok(_) => {
            println!("Deleted collection: {}", args.collection_name);
        }
        Err(e) => {
            println!("Failed to delete collection: {e:?}");
        }
    }

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    let datatype: Option<i32> = args
        .datatype
        .as_ref()
        .map(|datatype| match datatype.as_str() {
            "Uint8" => Datatype::Uint8.into(),
            "Float16" => Datatype::Float16.into(),
            "Float32" => Datatype::Float32.into(),
            "Turbo4" => Datatype::Turbo4.into(),
            _ => {
                panic!("Unknown vector datatype {datatype}")
            }
        });

    let multivector_config = args.multivector_size.map(|_multivector_size| {
        MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim).build()
    });

    let distance = match args.distance.as_str() {
        "Cosine" => Distance::Cosine,
        "Dot" => Distance::Dot,
        "Euclid" => Distance::Euclid,
        "Manhattan" => Distance::Manhattan,
        _ => {
            panic!("Unknown distance {}", args.distance)
        }
    };

    let mut vector_param_builder = VectorParamsBuilder::new(args.dim as u64, distance);
    if let Some(on_disk) = args.on_disk_vectors {
        vector_param_builder = vector_param_builder.on_disk(on_disk);
    }
    if let Some(memory) = args.memory_vectors {
        vector_param_builder = vector_param_builder.memory(memory_to_i32(memory.into()));
    }
    if let Some(multivector_config) = multivector_config {
        vector_param_builder = vector_param_builder.multivector_config(multivector_config);
    }
    if let Some(datatype) = datatype {
        vector_param_builder = vector_param_builder.datatype(datatype);
    }
    let vector_param = vector_param_builder.build();

    let dense_vector_params = if args.vectors_per_point == 1 {
        Config::Params(vector_param)
    } else {
        let params = (0..args.vectors_per_point)
            .map(|idx| (idx.to_string(), vector_param))
            .collect();

        Config::ParamsMap(VectorParamsMap { map: params })
    };

    let vectors_config: VectorsConfig = dense_vector_params.clone().into();

    let sparse_vectors_config = if args.sparse_vectors.is_some() {
        let params: HashMap<_, _> = (0..args.sparse_vectors_per_point)
            .map(|idx| {
                let key = format!("{idx}_sparse");
                let mut index_builder = SparseIndexConfigBuilder::default()
                    .on_disk(args.on_disk_index.unwrap_or_default());
                if let Some(datatype) = datatype {
                    index_builder = index_builder.datatype(datatype);
                }
                if let Some(memory) = args.memory_index {
                    index_builder = index_builder.memory(memory_to_i32(memory.into()));
                }
                let mut config = SparseVectorParamsBuilder::default().index(index_builder);
                if args.sparse_idf {
                    config = config.modifier(Modifier::Idf);
                }

                (key, config.build())
            })
            .collect();

        Some(SparseVectorConfig::from(params))
    } else {
        None
    };

    // Hnsw config
    let mut hnsw_config =
        HnswConfigDiffBuilder::default().on_disk(args.on_disk_index.unwrap_or_default());
    if let Some(m) = args.hnsw_m {
        hnsw_config = hnsw_config.m(m as u64);
    }
    if let Some(payload_m) = args.hnsw_payload_m {
        hnsw_config = hnsw_config.payload_m(payload_m as u64);
    }
    if let Some(ef_construct) = args.hnsw_ef_construct {
        hnsw_config = hnsw_config.ef_construct(ef_construct as u64);
    }
    if let Some(fs_th) = args.full_scan_threshold {
        hnsw_config = hnsw_config.full_scan_threshold(fs_th as u64);
    }
    if args.hnsw_inline_storage {
        hnsw_config = hnsw_config.inline_storage(true);
    }
    if let Some(memory) = args.memory_index {
        hnsw_config = hnsw_config.memory(memory_to_i32(memory.into()));
    }

    let mut optimizers_config = OptimizersConfigDiffBuilder::default();
    if let Some(default_segment_number) = args.segments {
        optimizers_config = optimizers_config.default_segment_number(default_segment_number as u64);
    }
    if let Some(mmap_threshold) = args.mmap_threshold {
        optimizers_config = optimizers_config.memmap_threshold(mmap_threshold as u64);
    }
    if let Some(indexing_threshold) = args.indexing_threshold {
        optimizers_config = optimizers_config.indexing_threshold(indexing_threshold as u64);
    }
    if let Some(max_segment_size) = args.max_segment_size {
        optimizers_config = optimizers_config.max_segment_size(max_segment_size as u64);
    }
    if args.prevent_unoptimized {
        optimizers_config = optimizers_config.prevent_unoptimized(true);
    }

    let mut create_collection_builder = CreateCollectionBuilder::new(args.collection_name.clone())
        .vectors_config(vectors_config)
        .hnsw_config(hnsw_config)
        .optimizers_config(optimizers_config)
        .on_disk_payload(args.on_disk_payload)
        .replication_factor(args.replication_factor as u32)
        .write_consistency_factor(args.write_consistency_factor as u32);

    if let Some(shard_number) = args.shards {
        create_collection_builder = create_collection_builder.shard_number(shard_number as u32);
    }

    if let Some(memory) = args.memory_payload {
        create_collection_builder = create_collection_builder
            .payload(PayloadStorageParamsBuilder::default().memory(memory_to_i32(memory.into())));
    }

    if let Some(sparse_vector_config) = sparse_vectors_config {
        create_collection_builder =
            create_collection_builder.sparse_vectors_config(sparse_vector_config);
    }

    if args.shard_key.is_some() {
        create_collection_builder =
            create_collection_builder.sharding_method(ShardingMethod::Custom.into());
    }

    if let Some(quantization) = args.quantization
        && let Some(quantization_config) = build_quantization_config(
            quantization.into(),
            args.quantization_in_ram.unwrap_or_default(),
            args.memory_quantization.map(Into::into),
        )
    {
        create_collection_builder =
            create_collection_builder.quantization_config(quantization_config);
    }

    client.create_collection(create_collection_builder).await?;
    println!("Created collection: {}", args.collection_name);

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if !args.skip_field_indices {
        create_field_indices(args, &client).await?;
    }

    if let Some(shard_key) = &args.shard_key {
        let mut builder = CreateShardKeyBuilder::default()
            .shard_key(Key::Keyword(shard_key.clone()))
            .replication_factor(args.replication_factor as u32);
        if let Some(shards) = args.shards {
            builder = builder.shards_number(shards as u32);
        }

        client
            .create_shard_key(
                CreateShardKeyRequestBuilder::new(args.collection_name.clone()).request(builder),
            )
            .await?;
    }

    Ok(())
}

async fn create_field_indices(args: &Args, client: &qdrant_client::Qdrant) -> Result<()> {
    // `memory` supersedes `on_disk` on servers that understand it; both are sent
    // so the same flags keep working against older ones.
    let memory = args.memory_payload_index.map(|m| memory_to_i32(m.into()));

    for (idx, _) in args.keywords.iter().enumerate() {
        let mut params = KeywordIndexParamsBuilder::default()
            .on_disk(args.on_disk_payload_index)
            .is_tenant(args.tenants.unwrap_or_default());
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), KEYWORD_PAYLOAD_KEY),
                    FieldType::Keyword,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    for (idx, _) in args.float_payloads.iter().enumerate() {
        let mut params = FloatIndexParamsBuilder::default()
            .on_disk(args.on_disk_payload_index)
            .is_principal(args.tenants.unwrap_or_default());
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), FLOAT_PAYLOAD_KEY),
                    FieldType::Float,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    for (idx, _) in args.int_payloads.iter().enumerate() {
        let mut params = IntegerIndexParamsBuilder::new(true, args.int_payloads_range)
            .on_disk(args.on_disk_payload_index)
            .is_principal(args.tenants.unwrap_or_default());
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), INTEGERS_PAYLOAD_KEY),
                    FieldType::Integer,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.timestamp_payload {
        let mut params = DatetimeIndexParamsBuilder::default()
            .on_disk(args.on_disk_payload_index)
            .is_principal(args.tenants.unwrap_or_default());
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    "timestamp",
                    FieldType::Datetime,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.uuid_payloads {
        let mut params = UuidIndexParamsBuilder::default()
            .is_tenant(args.tenants.unwrap_or_default())
            .on_disk(args.on_disk_payload_index);
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    UUID_PAYLOAD_KEY,
                    FieldType::Uuid,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.geo_payloads {
        let mut params = GeoIndexParamsBuilder::new().on_disk(args.on_disk_payload_index);
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    GEO_PAYLOAD_KEY,
                    FieldType::Geo,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.bool_payloads {
        let mut params = BoolIndexParamsBuilder::default().on_disk(args.on_disk_payload_index);
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    BOOL_PAYLOAD_KEY,
                    FieldType::Bool,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.text_payloads {
        let mut params =
            TextIndexParamsBuilder::new(TokenizerType::Word).on_disk(args.on_disk_payload_index);
        if let Some(memory) = memory {
            params = params.memory(memory);
        }
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    TEXT_PAYLOAD_KEY,
                    FieldType::Text,
                )
                .field_index_params(params)
                .wait(true),
            )
            .await
            .unwrap();
    }

    Ok(())
}
