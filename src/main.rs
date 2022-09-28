use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use clap::Parser;
use qdrant_client::client::{Payload, QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{CreateCollection, Distance, PointStruct, VectorParams, VectorsConfig, CollectionStatus};
use qdrant_client::qdrant::vectors_config::Config;
use tokio::runtime;
use tokio::time::sleep;
use anyhow::Result;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::Rng;


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

    /// If set, after upload will wait until collection is indexed
    #[clap(long, default_value_t = true)]
    wait_index: bool,

    #[clap(long, default_value = "benchmark")]
    collection_name: String,

    #[clap(long, default_value = "Cosine")]
    distance: String,

    /// Log requests if the take longer than this
    #[clap(long, default_value_t = 0.1)]
    timing_threshold: f64,
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

async fn wait_index(client: &QdrantClient, args: Args, stopped: Arc<AtomicBool>) -> Result<f64> {
    let start = std::time::Instant::now();
    let mut seen = 0;
    loop {
        if stopped.load(Ordering::Relaxed) {
            return Ok(0.0)
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

fn get_config(args: &Args) -> QdrantClientConfig {
    let mut config = QdrantClientConfig::from_url(&args.uri);
    let api_key = std::env::var("QDRANT_API_KEY").ok();

    if let Some(api_key) = api_key {
        config.set_api_key(&api_key);
    }
    config
}

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
    let client = QdrantClient::new(Some(get_config(&args))).await?;

    client.delete_collection(&args.collection_name).await?;
    sleep(Duration::from_secs(1)).await;
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

    sleep(Duration::from_secs(1)).await;

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
            points.push(PointStruct::new(
                if let Some(max_id) = args.max_id {
                    rng.gen_range(0..max_id) as u64
                } else {
                    n as u64
                },
                random_vector(args.dim),
                Payload::new(),
            ));
            n += 1;
        }

        if stopped.load(Ordering::Relaxed) {
            return Ok(())
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

    if args.wait_index {
        println!("Waiting for index to be ready...");
        let wait_time = wait_index(&client, args.clone(), stopped).await?;
        println!("Index ready in {} seconds", wait_time);
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
