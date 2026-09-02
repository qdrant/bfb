//! `bfb serverless query` — route searches across existing collections.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use qdrant_client::qdrant::{QueryPointsBuilder, VectorInput};
use rand::Rng;
use rand::RngExt;

use super::args::ServerlessQueryArgs;
use super::client::{self, retry_with_clients};
use super::collections::list_matching;
use super::distribution::CollectionPicker;
use crate::args::Args;
use crate::generators::random::random_dense_vector;
use crate::processor::Timing;
use crate::stats::{print_stats, throttler};
use qdrant_client::serverless::CollectionConfig;

/// Shape used when no search YAML is provided: taken from a live collection.
struct InferredShape {
    /// `(vector_name, size)` — empty name is the default unnamed vector.
    dense: Vec<(String, u64)>,
}

impl InferredShape {
    fn from_config(config: &CollectionConfig) -> Result<Self> {
        let mut dense: Vec<(String, u64)> = config
            .dense_vectors
            .iter()
            .map(|(name, cfg)| (name.clone(), cfg.size))
            .collect();
        dense.sort_by(|a, b| a.0.cmp(&b.0));
        if dense.is_empty() {
            bail!("collection has no dense vectors to query");
        }
        Ok(Self { dense })
    }

    fn random_query(&self, rng: &mut impl Rng) -> (Option<String>, Vec<f32>) {
        let idx = rng.random_range(0..self.dense.len());
        let (name, size) = &self.dense[idx];
        let vector = random_dense_vector(rng, *size as usize, false);
        let using = if name.is_empty() {
            None
        } else {
            Some(name.clone())
        };
        (using, vector)
    }
}

pub async fn run(args: &Args, query: ServerlessQueryArgs, stopped: Arc<AtomicBool>) -> Result<()> {
    let clients = client::create_clients(args)?;
    let names = list_matching(&clients[0], &query.collection_prefix).await?;
    if names.is_empty() {
        bail!(
            "no collections matched prefix {:?} — run `bfb serverless upload` first",
            query.collection_prefix
        );
    }

    println!(
        "Querying {} collection(s) with prefix {:?} ({:?})",
        names.len(),
        query.collection_prefix,
        query.distribution
    );

    // Guess vector shape from an existing collection's config when no search
    // YAML is given (Notion: "guess shape of the vectors from collection config").
    if query.config_file.is_some() {
        // Search-shape YAML for serverless is not wired yet; infer instead.
        eprintln!(
            "note: --config-file on `serverless query` is accepted but ignored for now; \
             vector shape is inferred from collection {:?}",
            names[0]
        );
    }
    let shape = infer_shape(&clients[0], &names[0]).await?;

    let picker = CollectionPicker::new(names.len(), query.distribution.into())?;
    let num_queries = args.num_vectors.unwrap_or(10_000);

    let logger = env_logger::Builder::from_default_env().build();
    let multiprogress = MultiProgress::new();
    indicatif_log_bridge::LogWrapper::new(multiprogress.clone(), logger)
        .try_init()
        .ok();

    let bar = multiprogress.add(ProgressBar::new(num_queries as u64));
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("progress style"),
    );
    bar.set_draw_target(ProgressDrawTarget::stdout_with_hz(2));
    let bar = Arc::new(bar);

    let server_timings = Arc::new(Mutex::new(Vec::<Timing>::new()));
    let full_timings = Arc::new(Mutex::new(Vec::<Timing>::new()));
    let rps_timings = Arc::new(Mutex::new(Vec::<Timing>::new()));
    let start = Instant::now();

    let parallel = if args.rps.is_some() {
        1
    } else {
        args.parallel.max(1)
    };
    let throttler = throttler(args.rps.map(|r| r as f32).or(args.throttle));
    let stopped_flag = stopped.clone();
    let clients = Arc::new(clients);
    let names = Arc::new(names);
    let shape = Arc::new(shape);
    let args_owned = args.clone();

    futures::stream::iter(0..num_queries)
        .take_while(|_| {
            let s = stopped_flag.clone();
            async move { !s.load(Ordering::Relaxed) }
        })
        .zip(throttler)
        .for_each_concurrent(parallel, |(req_id, _)| {
            let clients = clients.clone();
            let names = names.clone();
            let shape = shape.clone();
            let bar = bar.clone();
            let server_timings = server_timings.clone();
            let full_timings = full_timings.clone();
            let rps_timings = rps_timings.clone();
            let args = args_owned.clone();
            let stopped = stopped_flag.clone();
            let picker = picker.clone();

            async move {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }

                let mut rng = rand::rng();
                let coll_idx = picker.pick(&mut rng);
                let collection = names[coll_idx].clone();
                let (using, vector) = shape.random_query(&mut rng);

                let mut builder = QueryPointsBuilder::new(collection.clone())
                    .query(VectorInput::new_dense(vector))
                    .limit(args.search_limit as u64)
                    .with_payload(args.search_with_payload)
                    .with_vectors(args.search_with_vectors);
                if let Some(name) = using {
                    builder = builder.using(name);
                }
                if let Some(timeout) = args.timeout {
                    builder = builder.timeout(timeout as u64);
                }
                let request = builder.build();

                let req_start = Instant::now();
                let res = retry_with_clients(&clients, &args, |c| c.query(request.clone())).await;
                let full = req_start.elapsed().as_secs_f32();

                match res {
                    Ok(resp) => {
                        let delay = start.elapsed().as_millis() as u32;
                        server_timings.lock().unwrap().push(Timing {
                            delay_millis: delay,
                            value: resp.time as f32,
                        });
                        full_timings.lock().unwrap().push(Timing {
                            delay_millis: delay,
                            value: full,
                        });
                        // Instantaneous RPS estimate from inter-arrival of completions.
                        let elapsed = start.elapsed().as_secs_f32().max(1e-6);
                        rps_timings.lock().unwrap().push(Timing {
                            delay_millis: delay,
                            value: (req_id as f32 + 1.0) / elapsed,
                        });
                        if resp.time > args.timing_threshold {
                            bar.println(format!("Slow query on {collection}: {:?}", resp.time));
                        }
                        bar.inc(1);
                    }
                    Err(e) => {
                        bar.println(format!("query failed on {collection}: {e:?}"));
                        if !args.ignore_errors {
                            stopped.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
        .await;

    bar.finish_and_clear();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Serverless query finished in {elapsed:.2}s ({:.0} qps wall)",
        num_queries as f64 / elapsed.max(1e-9)
    );

    let mut server = server_timings.lock().unwrap().clone();
    let mut full = full_timings.lock().unwrap().clone();
    let mut rps = rps_timings.lock().unwrap().clone();
    print_stats(args, &mut server, "server time", true);
    print_stats(args, &mut full, "full time", true);
    print_stats(args, &mut rps, "rps", false);
    Ok(())
}

async fn infer_shape(
    client: &qdrant_client::serverless::QdrantServerless,
    name: &str,
) -> Result<InferredShape> {
    let info = client
        .get_collection(name)
        .await
        .with_context(|| format!("get_collection {name}"))?;
    if !info.exists {
        bail!("collection {name} no longer exists");
    }
    let config = info.config.ok_or_else(|| {
        anyhow::anyhow!("collection {name} has no config; re-upload or pass a search config")
    })?;
    InferredShape::from_config(&config)
}
