use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CollectionStatus, CompressionRatio, CreateCollection, Distance, FieldType, HnswConfigDiff,
    OptimizersConfigDiff, ProductQuantization, QuantizationConfig, QuantizationType,
    ScalarQuantization, SparseIndexConfig, SparseVectorConfig, SparseVectorParams, VectorParams,
    VectorParamsMap, VectorsConfig,
};
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

fn get_config(args: &Args) -> Vec<QdrantClientConfig> {
    let mut configs = Vec::new();

    for uri in args.uri.iter() {
        let mut config = QdrantClientConfig::from_url(uri);
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
    configs
}

async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let client = QdrantClient::new(Some(choose_owned(get_config(args))))?;
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
    let client = QdrantClient::new(Some(choose_owned(get_config(args))))?;

    if args.create_if_missing && client.collection_info(&args.collection_name).await.is_ok() {
        println!("Collection already exists");
        return Ok(());
    }

    match client.delete_collection(&args.collection_name).await {
        Ok(_) => {}
        Err(e) => {
            println!("Failed to delete collection: {}", e);
        }
    }

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    let _vector_param = VectorParams {
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
        Config::Params(_vector_param)
    } else {
        let params = (0..args.vectors_per_point)
            .map(|idx| (idx.to_string(), _vector_param.clone()))
            .collect();

        Config::ParamsMap(VectorParamsMap { map: params })
    };

    let vectors_config = Some(VectorsConfig {
        config: Some(dense_vector_params),
    });

    let sparse_vectors_config = if args.sparse_vectors.is_some() {
        let params = (0..args.sparse_vectors_per_point)
            .map(|idx| {
                (
                    format!("{idx}_sparse").to_string(),
                    SparseVectorParams {
                        index: Some(SparseIndexConfig {
                            full_scan_threshold: None,
                            on_disk: args.on_disk_index,
                        }),
                    },
                )
            })
            .collect();

        Some(SparseVectorConfig { map: params })
    } else {
        None
    };

    client
        .create_collection(&CreateCollection {
            collection_name: args.collection_name.clone(),
            vectors_config,
            hnsw_config: Some(HnswConfigDiff {
                on_disk: args.on_disk_index,
                m: args.hnsw_m.map(|x| x as u64),
                ef_construct: args.hnsw_ef_construct.map(|x| x as u64),
                ..Default::default()
            }),
            optimizers_config: Some(OptimizersConfigDiff {
                default_segment_number: args.segments.map(|x| x as u64),
                memmap_threshold: args.mmap_threshold.map(|x| x as u64),
                indexing_threshold: args.indexing_threshold.map(|x| x as u64),
                max_segment_size: args.max_segment_size.map(|x| x as u64),
                ..Default::default()
            }),
            on_disk_payload: Some(args.on_disk_payload),
            replication_factor: Some(args.replication_factor as u32),
            write_consistency_factor: Some(args.write_consistency_factor as u32),
            shard_number: args.shards.map(|x| x as u32),
            quantization_config: match args.quantization {
                Some(quantization) => match quantization {
                    QuantizationArg::None => None,
                    QuantizationArg::Scalar => Some(QuantizationConfig {
                        quantization: Some(Quantization::Scalar(ScalarQuantization {
                            r#type: i32::from(QuantizationType::Int8),
                            quantile: Some(0.99),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                    QuantizationArg::ProductX4 => Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: CompressionRatio::X4.into(),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                    QuantizationArg::ProductX8 => Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: CompressionRatio::X8.into(),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                    QuantizationArg::ProductX16 => Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: CompressionRatio::X16.into(),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                    QuantizationArg::ProductX32 => Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: CompressionRatio::X32.into(),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                    QuantizationArg::ProductX64 => Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: CompressionRatio::X64.into(),
                            always_ram: args.quantization_in_ram,
                        })),
                    }),
                },
                None => None,
            },
            sparse_vectors_config,
            ..Default::default()
        })
        .await?;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if !args.skip_field_indices {
        for (idx, _) in args.keywords.iter().enumerate() {
            client
                .create_field_index_blocking(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), KEYWORD_PAYLOAD_KEY),
                    FieldType::Keyword,
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        for (idx, _) in args.float_payloads.iter().enumerate() {
            client
                .create_field_index_blocking(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), FLOAT_PAYLOAD_KEY),
                    FieldType::Float,
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        for (idx, _) in args.int_payloads.iter().enumerate() {
            client
                .create_field_index_blocking(
                    args.collection_name.clone(),
                    format!("{}{}", payload_prefixes(idx), INTEGERS_PAYLOAD_KEY),
                    FieldType::Integer,
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        if args.timestamp_payload {
            client
                .create_field_index_blocking(
                    args.collection_name.clone(),
                    "timestamp",
                    FieldType::Datetime,
                    None,
                    None,
                )
                .await
                .unwrap();

        }
    }

    Ok(())
}

async fn upload_data(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(QdrantClient::new(Some(config))?);
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
        clients.push(QdrantClient::new(Some(config))?);
    }
    let searcher = SearchProcessor::new(args.clone(), stopped.clone(), clients);
    process(args, stopped, searcher).await
}

async fn scroll(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let mut clients = Vec::new();
    for config in get_config(args) {
        clients.push(QdrantClient::new(Some(config))?);
    }
    let searcher = ScrollProcessor::new(args.clone(), stopped.clone(), clients);
    process(args, stopped, searcher).await
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
