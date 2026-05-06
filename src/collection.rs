use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    BinaryQuantizationBuilder, BinaryQuantizationEncoding, BoolIndexParamsBuilder,
    CollectionStatus, CompressionRatio, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    CreateShardKeyBuilder, CreateShardKeyRequestBuilder, Datatype, DatetimeIndexParamsBuilder,
    Distance, FieldType, FloatIndexParamsBuilder, HnswConfigDiffBuilder, IntegerIndexParamsBuilder,
    KeywordIndexParamsBuilder, MultiVectorComparator, MultiVectorConfigBuilder,
    OptimizersConfigDiffBuilder, ProductQuantizationBuilder, QuantizationType,
    ScalarQuantizationBuilder, ShardingMethod, SparseIndexConfigBuilder, SparseVectorConfig,
    SparseVectorParamsBuilder, TextIndexParamsBuilder, TokenizerType, TurboQuantBitSize,
    TurboQuantizationBuilder, UuidIndexParamsBuilder, VectorParams, VectorParamsMap, VectorsConfig,
};
use tokio::time::sleep;

use crate::args::{Args, QuantizationArg};
use crate::client::random_client;
use crate::common::{
    BOOL_PAYLOAD_KEY, FLOAT_PAYLOAD_KEY, GEO_PAYLOAD_KEY, INTEGERS_PAYLOAD_KEY,
    KEYWORD_PAYLOAD_KEY, TEXT_PAYLOAD_KEY, UUID_PAYLOAD_KEY, payload_prefixes,
};

pub async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let client = random_client(args)?;
    let start = std::time::Instant::now();
    let mut seen = 0;
    loop {
        if stopped.load(Ordering::Relaxed) {
            return Ok(0.0);
        }
        sleep(Duration::from_secs(1)).await;
        let info = client.collection_info(&args.collection_name).await?;
        if info.result.unwrap().status == CollectionStatus::Green as i32 {
            seen += 1;
            if seen == 3 {
                break;
            }
        } else {
            seen = 1;
        }
    }
    Ok(start.elapsed().as_secs_f64())
}

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

    let datatype = args
        .datatype
        .as_ref()
        .map(|datatype| match datatype.as_str() {
            "Uint8" => Datatype::Uint8.into(),
            "Float16" => Datatype::Float16.into(),
            "Float32" => Datatype::Float32.into(),
            _ => {
                panic!("Unknown vector datatype {datatype}")
            }
        });

    let multivector_config = args.multivector_size.map(|_multivector_size| {
        MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim).build()
    });

    let vector_param = VectorParams {
        size: args.dim as u64,
        distance: match args.distance.as_str() {
            "Cosine" => Distance::Cosine.into(),
            "Dot" => Distance::Dot.into(),
            "Euclid" => Distance::Euclid.into(),
            "Manhattan" => Distance::Manhattan.into(),
            _ => {
                panic!("Unknown distance {}", args.distance)
            }
        },
        on_disk: args.on_disk_vectors,
        multivector_config,
        datatype,
        ..Default::default()
    };

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
                let config = SparseVectorParamsBuilder::default()
                    .index(index_builder)
                    .build();

                (key, config)
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

    if let Some(sparse_vector_config) = sparse_vectors_config {
        create_collection_builder =
            create_collection_builder.sparse_vectors_config(sparse_vector_config);
    }

    if args.shard_key.is_some() {
        create_collection_builder =
            create_collection_builder.sharding_method(ShardingMethod::Custom.into());
    }

    if let Some(quantization) = args.quantization {
        create_collection_builder = match quantization {
            QuantizationArg::Scalar => create_collection_builder.quantization_config(
                ScalarQuantizationBuilder::default()
                    .r#type(QuantizationType::Int8.into())
                    .quantile(0.99)
                    .always_ram(args.quantization_in_ram.unwrap_or_default()),
            ),
            QuantizationArg::None => create_collection_builder,
            QuantizationArg::Binary => create_collection_builder.quantization_config(
                BinaryQuantizationBuilder::new(args.quantization_in_ram.unwrap_or_default()),
            ),
            QuantizationArg::Binary2bit => create_collection_builder.quantization_config(
                BinaryQuantizationBuilder::new(args.quantization_in_ram.unwrap_or_default())
                    .encoding(BinaryQuantizationEncoding::TwoBits),
            ),
            QuantizationArg::Binary1p5bit => create_collection_builder.quantization_config(
                BinaryQuantizationBuilder::new(args.quantization_in_ram.unwrap_or_default())
                    .encoding(BinaryQuantizationEncoding::OneAndHalfBits),
            ),
            QuantizationArg::Turbo1Bit
            | QuantizationArg::Turbo1_5Bit
            | QuantizationArg::Turbo2Bit
            | QuantizationArg::Turbo4Bit => {
                let size = match quantization {
                    QuantizationArg::Turbo1Bit => TurboQuantBitSize::Bits1,
                    QuantizationArg::Turbo1_5Bit => TurboQuantBitSize::Bits15,
                    QuantizationArg::Turbo2Bit => TurboQuantBitSize::Bits2,
                    QuantizationArg::Turbo4Bit => TurboQuantBitSize::Bits4,
                    QuantizationArg::None
                    | QuantizationArg::Binary
                    | QuantizationArg::Binary2bit
                    | QuantizationArg::Binary1p5bit
                    | QuantizationArg::Scalar
                    | QuantizationArg::ProductX4
                    | QuantizationArg::ProductX8
                    | QuantizationArg::ProductX16
                    | QuantizationArg::ProductX32
                    | QuantizationArg::ProductX64 => unreachable!(),
                };

                let data_fit_disabled = args.turbo_quant_disable_data_fit == Some(true);

                let tq_builder = TurboQuantizationBuilder::new()
                    .always_ram(args.quantization_in_ram.unwrap_or_default())
                    .bits(size);

                create_collection_builder.quantization_config(if data_fit_disabled {
                    tq_builder.data_fit_disabled()
                } else {
                    tq_builder
                })
            }
            quantization => {
                let compression = match quantization {
                    QuantizationArg::ProductX4 => CompressionRatio::X4,
                    QuantizationArg::ProductX8 => CompressionRatio::X8,
                    QuantizationArg::ProductX16 => CompressionRatio::X16,
                    QuantizationArg::ProductX32 => CompressionRatio::X32,
                    QuantizationArg::ProductX64 => CompressionRatio::X64,
                    QuantizationArg::Scalar
                    | QuantizationArg::Binary
                    | QuantizationArg::Binary2bit
                    | QuantizationArg::Binary1p5bit
                    | QuantizationArg::None
                    | QuantizationArg::Turbo1Bit
                    | QuantizationArg::Turbo1_5Bit
                    | QuantizationArg::Turbo2Bit
                    | QuantizationArg::Turbo4Bit => {
                        unreachable!()
                    }
                };
                create_collection_builder.quantization_config(
                    ProductQuantizationBuilder::new(compression.into())
                        .always_ram(args.quantization_in_ram.unwrap_or_default()),
                )
            }
        };
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
    for (idx, _) in args.keywords.iter().enumerate() {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), KEYWORD_PAYLOAD_KEY),
                    FieldType::Keyword,
                )
                .field_index_params(
                    KeywordIndexParamsBuilder::default()
                        .on_disk(args.on_disk_payload_index)
                        .is_tenant(args.tenants.unwrap_or_default()),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    for (idx, _) in args.float_payloads.iter().enumerate() {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), FLOAT_PAYLOAD_KEY),
                    FieldType::Float,
                )
                .field_index_params(
                    FloatIndexParamsBuilder::default()
                        .on_disk(args.on_disk_payload_index)
                        .is_principal(args.tenants.unwrap_or_default()),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    for (idx, _) in args.int_payloads.iter().enumerate() {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), INTEGERS_PAYLOAD_KEY),
                    FieldType::Integer,
                )
                .field_index_params(
                    IntegerIndexParamsBuilder::new(true, args.int_payloads_range)
                        .on_disk(args.on_disk_payload_index)
                        .is_principal(args.tenants.unwrap_or_default()),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.timestamp_payload {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    "timestamp",
                    FieldType::Datetime,
                )
                .field_index_params(
                    DatetimeIndexParamsBuilder::default()
                        .is_principal(args.tenants.unwrap_or_default()),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.uuid_payloads {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    UUID_PAYLOAD_KEY,
                    FieldType::Uuid,
                )
                .field_index_params(
                    UuidIndexParamsBuilder::default()
                        .is_tenant(args.tenants.unwrap_or_default())
                        .on_disk(args.on_disk_payload_index),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.geo_payloads {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    GEO_PAYLOAD_KEY,
                    FieldType::Geo,
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.bool_payloads {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    BOOL_PAYLOAD_KEY,
                    FieldType::Bool,
                )
                .field_index_params(
                    BoolIndexParamsBuilder::default().on_disk(args.on_disk_payload_index),
                )
                .wait(true),
            )
            .await
            .unwrap();
    }

    if args.text_payloads {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    args.collection_name.clone(),
                    TEXT_PAYLOAD_KEY,
                    FieldType::Text,
                )
                .field_index_params(TextIndexParamsBuilder::new(TokenizerType::Word))
                .wait(true),
            )
            .await
            .unwrap();
    }

    Ok(())
}
