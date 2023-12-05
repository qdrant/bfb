use crate::args::Args;
use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::client::{Payload, QdrantClient};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{FieldCondition, Filter, Match, Range, Vector};
use rand::prelude::SliceRandom;
use rand::Rng;

pub const KEYWORD_PAYLOAD_KEY: &str = "a";
pub const FLOAT_PAYLOAD_KEY: &str = "b";

pub const INTEGERS_PAYLOAD_KEY: &str = "c";

pub fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::thread_rng();
    let variant = rng.gen_range(0..num_variants);
    format!("keyword_{}", variant)
}

pub fn random_payload(
    keywords: Option<usize>,
    float_payloads: bool,
    integer_payload: Option<usize>,
) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords {
        payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(keyword_variants));
    }
    if float_payloads {
        payload.insert(FLOAT_PAYLOAD_KEY, rand::thread_rng().gen_range(-1.0..1.0));
    }

    if let Some(integer_range) = integer_payload {
        let value = rand::thread_rng().gen_range(0..integer_range);
        payload.insert(INTEGERS_PAYLOAD_KEY, value as i64);
    }

    payload
}

pub fn random_filter(
    keywords: Option<usize>,
    float_payloads: bool,
    integer_payload: Option<usize>,
) -> Option<Filter> {
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
    };
    let mut have_any = false;
    if let Some(keyword_variants) = keywords {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: KEYWORD_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Keyword(random_keyword(keyword_variants))),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                geo_polygon: None,
                values_count: None,
            }
            .into(),
        )
    }
    if float_payloads {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: FLOAT_PAYLOAD_KEY.to_string(),
                r#match: None,
                range: Some(Range {
                    gt: Some(0.0),
                    gte: None,
                    lt: None,
                    lte: None,
                }),
                geo_bounding_box: None,
                geo_radius: None,
                geo_polygon: None,
                values_count: None,
            }
            .into(),
        )
    }
    if let Some(integer_range) = integer_payload {
        have_any = true;
        let rand_int = rand::thread_rng().gen_range(0..integer_range);
        filter.must.push(
            FieldCondition {
                key: INTEGERS_PAYLOAD_KEY.to_string(),
                r#match: None,
                range: Some(Range {
                    gt: Some(rand_int as f64),
                    gte: None,
                    lt: None,
                    lte: None,
                }),
                geo_bounding_box: None,
                geo_radius: None,
                geo_polygon: None,
                values_count: None,
            }
            .into(),
        )
    }

    have_any.then_some(filter)
}

pub fn random_vector(args: &Args) -> Vector {
    if args.sparse_vectors {
        random_sparse_vector(args.dim).into()
    } else {
        random_dense_vector(args.dim).into()
    }
}

/// Generate random sparse vector with random size and random values.
/// - `max_size` - maximum size of vector
pub fn random_sparse_vector(max_size: usize) -> Vec<(u32, f32)> {
    let mut rng = rand::thread_rng();
    let size = rng.gen_range(1..max_size);
    // (index, value)
    let mut pairs = Vec::with_capacity(size);
    for i in 1..=size {
        // probability of skipping a dimension to make the vectors sparse
        let skip = rng.gen_bool(0.1);
        if skip {
            continue;
        }
        // Only positive values are generated to make sure to hit the pruning path.
        pairs.push((i as u32, rng.gen_range(0.0..10.0) as f32));
    }
    pairs
}

pub fn random_dense_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

pub fn random_vector_name(max: usize) -> String {
    let mut rng = rand::thread_rng();
    format!("{}", rng.gen_range(0..max))
}

pub async fn retry_with_clients<'a, R, T: std::future::Future<Output = anyhow::Result<R>>>(
    clients: &'a [QdrantClient],
    mut call: impl FnMut(&'a QdrantClient) -> T,
) -> anyhow::Result<R> {
    let mut permutation = (0..clients.len()).collect::<Vec<_>>();
    permutation.shuffle(&mut rand::thread_rng());
    let mut res = Err(anyhow::anyhow!("No clients"));
    for client_id in permutation {
        let client = clients.get(client_id).unwrap();

        res = call(client).await;

        if res.is_ok() {
            break;
        }
    }
    res
}
