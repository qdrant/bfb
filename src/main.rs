use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use clap::Parser;
use qdrant_client::client::{Payload, QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{CreateCollection, Distance, PointStruct, VectorParams, VectorsConfig, CollectionStatus, PointId, FieldType, Filter, FieldCondition, Match, SearchPoints};
use qdrant_client::qdrant::vectors_config::Config;
use tokio::runtime;
use tokio::time::sleep;
use anyhow::Result;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::qdrant::r#match::MatchValue;
use rand::Rng;

const KEYWORD_PAYLOAD_KEY: &str = "a";

/// Big Fucking Benchmark tool for stress-testing Qdrant
#[derive(Parser, Debug, Clone)]
#[clap(version, about)]
struct Args {
    /// Qdrant service URI
    #[clap(long, default_value = "http://localhost:6334")]
    uri: String,

    #[clap(short, long, default_value_t = 100_000)]
    num_vectors: usize,

    /// If set, will use vector ids within range [0, max_id)
    /// To simulate overwriting existing vectors
    #[clap(short, long)]
    max_id: Option<usize>,

    #[clap(short, long, default_value_t = 128)]
    dim: usize,

    #[clap(short, long, default_value_t = 2)]
    threads: usize,

    /// Number of parallel requests to send
    #[clap(short, long, default_value_t = 2)]
    parallel: usize,

    #[clap(short, long, default_value_t = 100)]
    batch_size: usize,

    /// Skip creation of the collection
    #[clap(long, default_value_t = false)]
    skip_create: bool,

    /// If set, after upload will wait until collection is indexed
    #[clap(long, default_value_t = false)]
    skip_wait_index: bool,

    /// Perform data upload
    #[clap(long, default_value_t = false)]
    skip_upload: bool,

    /// Perform search
    #[clap(long, default_value_t = false)]
    search: bool,

    /// Search limit
    #[clap(long, default_value_t = 10)]
    search_limit: usize,

    #[clap(long, default_value = "benchmark")]
    collection_name: String,

    #[clap(long, default_value = "Cosine")]
    distance: String,

    /// Log requests if the take longer than this
    #[clap(long, default_value_t = 0.1)]
    timing_threshold: f64,

    /// Use UUIDs instead of sequential ids
    #[clap(long, default_value_t = false)]
    uuids: bool,

    /// Use keyword payloads. Defines how many different keywords there are in the payload
    #[clap(long)]
    keywords: Option<usize>,
}

fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::thread_rng();
    let variant = rng.gen_range(0..num_variants);
    format!("keyword_{}", variant)
}

fn random_payload(keywords: Option<usize>) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords {
        payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(keyword_variants));
    }
    payload
}

fn random_filter(keywords: Option<usize>) -> Option<Filter> {
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
    };
    let mut have_any = false;
    if let Some(keyword_variants) = keywords {
        have_any = true;
        filter.must.push(FieldCondition {
            key: KEYWORD_PAYLOAD_KEY.to_string(),
            r#match: Some(Match {
                match_value: Some(MatchValue::Keyword(random_keyword(keyword_variants))),
            }),
            range: None,
            geo_bounding_box: None,
            geo_radius: None,
            values_count: None,
        }.into())
    }
    if have_any {
        Some(filter)
    } else {
        None
    }
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn get_config(args: &Args) -> QdrantClientConfig {
    let mut config = QdrantClientConfig::from_url(&args.uri);
    let api_key = std::env::var("QDRANT_API_KEY").ok();

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

    client.create_collection(&CreateCollection {
        collection_name: args.collection_name.clone(),
        vectors_config: Some(
            VectorsConfig {
                config: Some(
                    Config::Params(
                        VectorParams {
                            size: args.dim as u64,
                            distance: match args.distance.as_str() {
                                "Cosine" => Distance::Cosine.into(),
                                "Dot" => Distance::Dot.into(),
                                "Euclid" => Distance::Euclid.into(),
                                _ => { panic!("Unknown distance {}", args.distance) }
                            },
                        }
                    )
                )
            }
        ),
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
    let recv_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    sent_bar.set_style(progress_style.clone());
    recv_bar.set_style(progress_style);

    let mut n = 0;
    let mut futures = FuturesUnordered::new();
    // let mut futures = Vec::new();
    let mut rng = rand::thread_rng();

    while n < args.num_vectors {
        let mut points = Vec::new();
        for _ in 0..args.batch_size {
            let idx = if let Some(max_id) = args.max_id {
                rng.gen_range(0..max_id) as u64
            } else {
                n as u64
            };

            let point_id: PointId = PointId {
                point_id_options: Some(if args.uuids {
                    PointIdOptions::Uuid(uuid::Uuid::from_u128(idx as u128).to_string())
                } else {
                    PointIdOptions::Num(idx)
                })
            };

            points.push(PointStruct::new(
                point_id,
                random_vector(args.dim),
                random_payload(args.keywords),
            ));
            n += 1;
        }

        if stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        futures.push(async {
            let batch_size = points.len() as u64;
            sent_bar.inc(batch_size);
            let res = client.upsert_points(&args.collection_name, points).await?;
            if res.time > args.timing_threshold {
                println!("Slow upsert: {:?}", res.time);
            }
            recv_bar.inc(batch_size);
            Ok(())
        });


        if futures.len() > args.parallel {
            let res: Result<_> = futures.next().await.unwrap();
            res?;
        }
    }

    while let Some(result) = futures.next().await {
        result?;
    }

    sent_bar.finish();
    recv_bar.finish();

    Ok(())
}

async fn search(args: &Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = QdrantClient::new(Some(get_config(args))).await?;

    let multiprogress = MultiProgress::new();

    let progress_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    progress_bar.set_style(progress_style);

    let timings = Mutex::new(Vec::new());
    let mut n = 0;
    let mut futures = FuturesUnordered::new();

    while n < args.num_vectors {
        let query_vector = random_vector(args.dim);
        let query_filter = random_filter(args.keywords);
        futures.push(async {
            let res = client.search_points(&SearchPoints {
                collection_name: args.collection_name.to_string(),
                vector: query_vector,
                filter: query_filter,
                limit: args.search_limit as u64,
                with_payload: Some(true.into()),
                params: None,
                score_threshold: None,
                offset: None,
                vector_name: None,
                with_vectors: None,
            }).await?;
            timings.lock().unwrap().push(res.time);

            if res.time > args.timing_threshold {
                println!("Slow search: {:?}", res.time);
            }
            progress_bar.inc(1);
            Ok(())
        });

        if stopped.load(Ordering::Relaxed) {
            return Ok(());
        }

        if futures.len() > args.parallel {
            let res: Result<_> = futures.next().await.unwrap();
            res?;
        }
        n += 1;
    }

    while let Some(result) = futures.next().await {
        result?;
    }

    progress_bar.finish();

    let mut timings = timings.lock().unwrap();

    timings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_time: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let max_time: f64 = timings.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
    let p95_time: f64 = timings[(timings.len() as f32 * 0.95) as usize];
    let p99_time: f64 = timings[(timings.len() as f32 * 0.99) as usize];

    println!("Avg search time: {}", avg_time);
    println!("p95 search time: {}", p95_time);
    println!("p99 search time: {}", p99_time);
    println!("Max search time: {}", max_time);

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
        let wait_time = wait_index( &args, stopped.clone()).await?;
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
