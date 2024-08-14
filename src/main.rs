use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use common::UUID_PAYLOAD_KEY;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CollectionStatus, CompressionRatio, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    CreateShardKeyBuilder, CreateShardKeyRequestBuilder, DatetimeIndexParamsBuilder, Distance,
    FieldType, FloatIndexParamsBuilder, HnswConfigDiffBuilder, IntegerIndexParamsBuilder,
    KeywordIndexParamsBuilder, OptimizersConfigDiffBuilder, ProductQuantizationBuilder,
    QuantizationType, ScalarQuantizationBuilder, ScrollPointsBuilder, ShardingMethod,
    SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParamsBuilder,
    UuidIndexParamsBuilder, VectorParams, VectorParamsMap, VectorsConfig,
};
use qdrant_client::Qdrant;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tokio::{join, runtime};

use args::Args;

use crate::args::QuantizationArg;
use crate::common::{
    payload_prefixes, random_dense_vector, random_filter, random_payload, throttler, Timing,
    FLOAT_PAYLOAD_KEY, INTEGERS_PAYLOAD_KEY, KEYWORD_PAYLOAD_KEY,
};
use crate::fbin_reader::FBinReader;
use crate::processor::Processor;
use crate::save_jsonl::save_timings_as_jsonl;
use crate::scroll::ScrollProcessor;
use crate::search::SearchProcessor;
use crate::upsert::UpsertProcessor;

mod args;
mod common;
mod fbin_reader;
mod processor;
mod save_jsonl;
mod scroll;
mod search;
mod upsert;

fn choose_owned<T>(mut items: Vec<T>) -> T {
    let mut rng = rand::thread_rng();
    // Get random id
    let id = rng.gen_range(0..items.len());
    // Remove item from vector
    items.swap_remove(id)
}

fn get_config(args: &Args) -> Vec<QdrantConfig> {
    let mut configs = Vec::new();

    for _i in 0..args.connections {
        for uri in args.uri.iter() {
            let mut config = QdrantConfig::from_url(uri);
            let api_key = std::env::var("QDRANT_API_KEY").ok();

            if let Some(timeout) = args.timeout {
                config.set_timeout(Duration::from_secs(timeout as u64));
                config.set_connect_timeout(Duration::from_secs(timeout as u64));
            }

            if let Some(api_key) = api_key {
                config.set_api_key(&api_key);
            }
            configs.push(config);
        }
    }
    configs
}

fn random_client(args: &Args) -> Result<Qdrant> {
    Ok(Qdrant::new(choose_owned(get_config(args)))?)
}

async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
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

async fn recreate_collection(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = random_client(args)?;

    if args.create_if_missing && client.collection_info(&args.collection_name).await.is_ok() {
        println!("Collection already exists");
        return Ok(());
    }

    match client.delete_collection(&args.collection_name).await {
        Ok(_) => {}
        Err(e) => {
            println!("Failed to delete collection: {:?}", e);
        }
    }

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    let vector_param = VectorParams {
        size: args.dim as u64,
        distance: match args.distance.as_str() {
            "Cosine" => Distance::Cosine.into(),
            "Dot" => Distance::Dot.into(),
            "Euclid" => Distance::Euclid.into(),
            _ => {
                panic!("Unknown distance {}", args.distance)
            }
        },
        on_disk: args.on_disk_vectors,
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

                let config = SparseVectorParamsBuilder::default()
                    .index(
                        SparseIndexConfigBuilder::default()
                            .on_disk(args.on_disk_index.unwrap_or_default()),
                    )
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
    if let Some(ef_construct) = args.hnsw_ef_construct {
        hnsw_config = hnsw_config.ef_construct(ef_construct as u64);
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
        if matches!(quantization, QuantizationArg::Scalar) {
            create_collection_builder = create_collection_builder.quantization_config(
                ScalarQuantizationBuilder::default()
                    .r#type(QuantizationType::Int8.into())
                    .quantile(0.99)
                    .always_ram(args.quantization_in_ram.unwrap_or_default()),
            );
        }

        if !matches!(quantization, QuantizationArg::None) {
            let compression = match quantization {
                QuantizationArg::ProductX4 => CompressionRatio::X4,
                QuantizationArg::ProductX8 => CompressionRatio::X8,
                QuantizationArg::ProductX16 => CompressionRatio::X16,
                QuantizationArg::ProductX32 => CompressionRatio::X32,
                QuantizationArg::ProductX64 => CompressionRatio::X64,
                QuantizationArg::Scalar | QuantizationArg::None => unreachable!(),
            };
            create_collection_builder = create_collection_builder.quantization_config(
                ProductQuantizationBuilder::new(compression.into())
                    .always_ram(args.quantization_in_ram.unwrap_or_default()),
            )
        }
    }

    client.create_collection(create_collection_builder).await?;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if !args.skip_field_indices {
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
                            .on_disk(args.on_disk_payload_index.unwrap_or_default())
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
                            .on_disk(args.on_disk_payload_index.unwrap_or_default())
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
                        IntegerIndexParamsBuilder::new(true, false)
                            .on_disk(args.on_disk_payload_index.unwrap_or_default())
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
                            .on_disk(args.on_disk_payload_index.unwrap_or_default()),
                    )
                    .wait(true),
                )
                .await
                .unwrap();
        }
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

async fn upload_data(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(Qdrant::new(config)?);
    }

    let logger = env_logger::Builder::from_default_env().build();

    let multiprogress = MultiProgress::new();

    indicatif_log_bridge::LogWrapper::new(multiprogress.clone(), logger)
        .try_init()
        .unwrap();

    let sent_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    sent_bar.set_style(progress_style);

    let sent_bar_arc = Arc::new(sent_bar);

    let reader = if let Some(path) = &args.fbin.as_ref() {
        FBinReader::new(Path::new(path)).into()
    } else {
        None
    };
    let upserter = UpsertProcessor::new(
        args.clone(),
        stopped.clone(),
        clients,
        sent_bar_arc.clone(),
        reader,
    );

    let num_batches = args.num_vectors / args.batch_size;

    let query_stream = (0..num_batches)
        .take_while(|_| !stopped.load(Ordering::Relaxed))
        .map(|n| {
            let future = upserter.upsert(n, args);
            sent_bar_arc.inc(args.batch_size as u64);
            future
        });

    if stopped.load(Ordering::Relaxed) {
        sent_bar_arc.abandon();
        return Ok(());
    }

    let mut throttler = throttler(args.throttle);
    let mut upsert_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let (Some(()), Some(result)) = { join!(throttler.next(), upsert_stream.next()) } {
        result?;
    }
    if stopped.load(Ordering::Relaxed) {
        sent_bar_arc.abandon();
    } else {
        sent_bar_arc.finish();
    }

    upserter.save_data().await;

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct SearcherResults {
    server_timings: Vec<f64>,
    rps: Vec<f64>,
    full_timings: Vec<f64>,
}

fn write_to_json(path: &String, results: SearcherResults) {
    let mut file = File::create(path).unwrap();
    serde_json::to_writer(&mut file, &results).unwrap();
    println!("Search results written to {}", path);
}

fn print_stats(args: &Args, values: &mut [Timing], metric_name: &str, show_percentiles: bool) {
    if values.is_empty() {
        return;
    }
    // sort values in ascending order
    values.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

    let avg_time: f64 = values.iter().map(|x| x.value).sum::<f64>() / values.len() as f64;
    let min_time: f64 = values.first().unwrap().value;
    let max_time: f64 = values.last().unwrap().value;
    let p50_time: f64 = values[(values.len() as f32 * 0.50) as usize].value;

    println!("Min {metric_name}: {min_time}");
    println!("Avg {metric_name}: {avg_time}");
    println!("Median {metric_name}: {p50_time}");

    if show_percentiles {
        let p95_time: f64 = values[(values.len() as f32 * 0.95) as usize].value;
        println!("p95 {metric_name}: {p95_time}");

        for digits in 2..=args.p9 {
            let factor = 1.0 - 1.0 * 0.1f64.powf(digits as f64);
            let index = ((values.len() as f64 * factor) as usize).min(values.len() - 1);
            let nines = "9".repeat(digits);
            let time = values[index].value;
            println!("p{nines} {metric_name}: {time}");
        }
    }

    println!("Max {metric_name}: {max_time}");
}

async fn process<P: Processor>(args: &Args, stopped: Arc<AtomicBool>, processor: P) -> Result<()> {
    let multiprogress = MultiProgress::new();
    let progress_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));
    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    progress_bar.set_style(progress_style);

    let query_stream = (0..args.num_vectors)
        .take_while(|_| !stopped.load(Ordering::Relaxed))
        .map(|n| {
            let future = processor.make_request(n, args, &progress_bar);
            progress_bar.inc(1);
            future
        });

    let mut throttler = throttler(args.throttle);
    let mut search_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let (Some(()), Some(result)) = { join!(throttler.next(), search_stream.next()) } {
        // Continue with no error
        let err = match result {
            Ok(()) => continue,
            Err(err) => err,
        };

        if args.ignore_errors {
            progress_bar.println(format!("Error: {}", err));
        } else {
            return Err(err);
        }
    }

    if stopped.load(Ordering::Relaxed) {
        progress_bar.abandon();
    } else {
        progress_bar.finish();
    }

    let mut full_timings = processor.full_timings();
    println!("--- Request timings ---");
    print_stats(args, &mut full_timings, "request time", true);
    let mut server_timings = processor.server_timings();
    println!("--- Server timings ---");
    print_stats(args, &mut server_timings, "server time", true);
    let mut rps = processor.rps();
    println!("--- RPS ---");
    print_stats(args, &mut rps, "rps", false);

    if args.json.is_some() {
        println!("--- Writing results to json file ---");
        write_to_json(
            args.json.as_ref().unwrap(),
            SearcherResults {
                server_timings: server_timings.iter().map(|x| x.value).collect(),
                rps: rps.iter().map(|x| x.value).collect(),
                full_timings: full_timings.iter().map(|x| x.value).collect(),
            },
        );
    }

    if let Some(jsonl_path) = &args.jsonl_searches {
        save_timings_as_jsonl(
            jsonl_path,
            args.absolute_time.unwrap_or(false),
            &server_timings,
            processor.start_timestamp_millis(),
            "request_latency",
        )?;
    }

    if let Some(jsonl_path) = &args.jsonl_rps {
        save_timings_as_jsonl(
            jsonl_path,
            args.absolute_time.unwrap_or(false),
            &rps,
            processor.start_timestamp_millis(),
            "request_rps",
        )?;
    }

    Ok(())
}

async fn search(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(Qdrant::new(config)?);
    }

    let uuids = get_uuids(args, &clients[0]).await?;

    let searcher = SearchProcessor::new(args.clone(), stopped.clone(), clients, uuids);
    process(args, stopped, searcher).await
}

/// If we want to retrieve points by UUIDs, we need to know about the existing UUIDs.
/// Here we decide which UUIDs we want to use for searching, based on the users preference.
async fn get_uuids(args: &Args, client: &Qdrant) -> Result<Vec<String>> {
    // Only use the UUID the user specified
    if let Some(uuid_query) = &args.uuid_query {
        return Ok(vec![uuid_query.to_string()]);
    }

    if !args.uuid_payloads {
        return Ok(vec![]);
    }
    
    // Retrieve existing UUIDs
    let res = client
        .scroll(
            ScrollPointsBuilder::new(&args.collection_name)
                .with_payload(true)
                .limit(args.num_vectors as u32),
        )
        .await?;
    let uuids: Vec<_> = res
        .result
        .iter()
        .filter_map(|i| {
            i.payload
                .get(UUID_PAYLOAD_KEY)
                .and_then(|j| j.as_str().map(|i| i.to_string()))
        })
        .collect();
    let uuids_count = uuids.len();
    let unique: HashSet<_> = uuids.into_iter().collect();
    if unique.len() != uuids_count {
        println!("Set of uuids not unique!");
    }

    // Make order random to not request the first point by its UUID.
    Ok(unique.into_iter().collect())
}

async fn scroll(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(Qdrant::new(config)?);
    }

    let uuids = get_uuids(args, &clients[0]).await?;

    let scroller = ScrollProcessor::new(args.clone(), stopped.clone(), clients, uuids);
    process(args, stopped, scroller).await
}

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
    if !args.skip_create {
        recreate_collection(&args, stopped.clone()).await?;
    }

    if !args.skip_upload {
        upload_data(&args, stopped.clone()).await?;
    }

    if !args.skip_wait_index {
        println!("Waiting for index to be ready...");
        let wait_time = wait_index(&args, stopped.clone()).await?;
        println!("Index ready in {} seconds", wait_time);
    }

    if args.search {
        search(&args, stopped.clone()).await?;
    }

    if args.scroll {
        scroll(&args, stopped.clone()).await?;
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        r.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build();

    runtime
        .unwrap()
        .block_on(run_benchmark(args, stopped))
        .unwrap();
}
