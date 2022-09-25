use std::time::Duration;
use clap::Parser;
use futures::future::try_join_all;
use qdrant_client::client::{Payload, QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{CreateCollection, Distance, PointStruct, VectorParams, VectorsConfig, CollectionStatus};
use qdrant_client::qdrant::vectors_config::Config;
use tokio::runtime;
use tokio::time::sleep;
use anyhow::Result;
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

async fn wait_index(client: &QdrantClient, args: Args) -> Result<f64> {
    let start = std::time::Instant::now();
    let mut seen = 0;
    loop {
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

async fn run_benchmark(args: Args) -> Result<()> {
    let config = QdrantClientConfig::from_url(&args.uri);
    let client = QdrantClient::new(Some(config)).await?;

    client.delete_collection(&args.collection_name).await?;
    sleep(Duration::from_secs(1)).await;
    client.create_collection(CreateCollection {
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

    let multiprogress = MultiProgress::new();

    let sent_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));
    let recv_bar = multiprogress.add(ProgressBar::new(args.num_vectors as u64));

    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] {wide_bar} {pos}/{len} (eta:{eta})")
        .expect("Failed to create progress style");
    sent_bar.set_style(progress_style.clone());
    recv_bar.set_style(progress_style);

    let mut n = 0;
    let mut futures = Vec::new();
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
        futures.push(async {
            sent_bar.inc(1);
            let res = client.upsert_points(&args.collection_name, points).await?;
            if res.time > args.timing_threshold {
                println!("Slow upsert: {:?}", res.time);
            }
            recv_bar.inc(1);
            Ok(())
        });
    }

    let res: Result<Vec<_>> = try_join_all(futures).await;

    res.unwrap();
    sent_bar.finish();
    recv_bar.finish();

    if args.wait_index {
        println!("Waiting for index to be ready...");
        let wait_time = wait_index(&client, args.clone()).await?;
        println!("Index ready in {} seconds", wait_time);
    }

    Ok(())
}


fn main() {
    let args = Args::parse();

    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build();

    runtime.unwrap().block_on(run_benchmark(args)).unwrap();
}
