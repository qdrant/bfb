mod common;
mod upsert;
mod search;
mod args;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{CollectionStatus, CreateCollection, Distance, FieldType, VectorParams, VectorParamsMap, VectorsConfig};
use qdrant_client::qdrant::vectors_config::Config;
use tokio::runtime;
use tokio::time::sleep;
use anyhow::Result;
use clap::Parser;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use args::Args;
use crate::common::{KEYWORD_PAYLOAD_KEY, random_filter, random_payload, random_vector};
use crate::search::SearchProcessor;
use crate::upsert::UpsertProcessor;

fn get_config(args: &Args) -> QdrantClientConfig {
    let mut config = QdrantClientConfig::from_url(&args.uri);
    let api_key = std::env::var("QDRANT_API_KEY").ok();

    if let Some(timeout) = args.timeout {
        config.set_timeout(Duration::from_secs(timeout as u64));
        config.set_connect_timeout(Duration::from_secs(timeout as u64));
    }

    if let Some(api_key) = api_key {
        config.set_api_key(&api_key);
    }
    config
}

async fn wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let client = QdrantClient::new(Some(get_config(args))).await?;
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
    let client = QdrantClient::new(Some(get_config(args))).await?;

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
            _ => { panic!("Unknown distance {}", args.distance) }
        },
    };

    let vector_params = if args.vectors_per_point == 1 {
        Config::Params(_vector_param)
    } else {
        let params = (0..args.vectors_per_point)
            .map(|idx| (
                idx.to_string(), _vector_param.clone()
            ))
            .collect();

        Config::ParamsMap(VectorParamsMap {
            map: params,
        })
    };

    client.create_collection(&CreateCollection {
        collection_name: args.collection_name.clone(),
        vectors_config: Some(
            VectorsConfig {
                config: Some(vector_params)
            }
        ),
        replication_factor: Some(args.replication_factor as u32),
        shard_number: args.shards.map(|x| x as u32),
        ..Default::default()
    }).await?;

    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    sleep(Duration::from_secs(1)).await;

    if args.keywords.is_some() {
        client.create_field_index_blocking(
            args.collection_name.clone(),
            KEYWORD_PAYLOAD_KEY,
            FieldType::Keyword,
            None,
        ).await.unwrap();
    }
    Ok(())
}

async fn upload_data(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = QdrantClient::new(Some(get_config(args))).await?;

    let multiprogress = MultiProgress::new();

    let sent_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    sent_bar.set_style(progress_style);

    let sent_bar_arc = Arc::new(sent_bar);
    let upserter = UpsertProcessor::new(args.clone(), stopped.clone(), client, sent_bar_arc.clone());

    let num_batches = args.num_vectors / args.batch_size;

    let query_stream = (0..num_batches).take_while(|_| !stopped.load(Ordering::Relaxed)).map(|n| {
        let future = upserter.upsert(n);
        sent_bar_arc.inc(args.batch_size as u64);
        future
    });

    let mut upsert_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let Some(result) = upsert_stream.next().await {
        result?;
    }
    if stopped.load(Ordering::Relaxed) {
        sent_bar_arc.abandon();
    } else {
        sent_bar_arc.finish();
    }

    Ok(())
}

fn print_timings(timings: &mut Vec<f64>) {
    if timings.is_empty() {
        return;
    }
    // sort timings in ascending order
    timings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_time: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let min_time: f64 = timings.first().copied().unwrap_or(0.0);
    let max_time: f64 = timings.last().copied().unwrap_or(0.0);
    let p95_time: f64 = timings[(timings.len() as f32 * 0.95) as usize];
    let p99_time: f64 = timings[(timings.len() as f32 * 0.99) as usize];

    println!("Min search time: {}", min_time);
    println!("Avg search time: {}", avg_time);
    println!("p95 search time: {}", p95_time);
    println!("p99 search time: {}", p99_time);
    println!("Max search time: {}", max_time);
}

async fn search(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = QdrantClient::new(Some(get_config(args))).await?;

    let searcher = SearchProcessor::new(args.clone(), stopped.clone(), client);

    let multiprogress = MultiProgress::new();
    let progress_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));
    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    progress_bar.set_style(progress_style);

    let query_stream = (0..args.num_vectors).take_while(|_| !stopped.load(Ordering::Relaxed)).map(|n| {
        let future = searcher.search(n, &progress_bar);
        progress_bar.inc(1);
        future
    });

    let mut search_stream = futures::stream::iter(query_stream).buffer_unordered(args.parallel);
    while let Some(result) = search_stream.next().await {
        result?;
    }

    if stopped.load(Ordering::Relaxed) {
        progress_bar.abandon();
    } else {
        progress_bar.finish();
    }

    let mut timings = searcher.full_timings.lock().unwrap();
    println!("--- Search timings ---");
    print_timings(&mut timings);
    let mut timings = searcher.server_timings.lock().unwrap();
    println!("--- Server timings ---");
    print_timings(&mut timings);

    Ok(())
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
    Ok(())
}


fn main() {
    let args = Args::parse();

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        r.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build();

    runtime.unwrap().block_on(run_benchmark(args, stopped)).unwrap();
}
