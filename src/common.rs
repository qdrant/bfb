use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::client::{Payload, QdrantClient};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{FieldCondition, Filter, Match, Range};
use rand::prelude::SliceRandom;
use rand::Rng;

pub const KEYWORD_PAYLOAD_KEY: &str = "a";
pub const FLOAT_PAYLOAD_KEY: &str = "b";

pub fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::thread_rng();
    let variant = rng.gen_range(0..num_variants);
    format!("keyword_{}", variant)
}

pub fn random_payload(keywords: Option<usize>, float_payloads: bool) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords {
        payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(keyword_variants));
    }
    if float_payloads {
        payload.insert(FLOAT_PAYLOAD_KEY, rand::thread_rng().gen_range(-1.0..1.0));
    }
    payload
}

pub fn random_filter(keywords: Option<usize>, float_payloads: bool) -> Option<Filter> {
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
                values_count: None,
            }
            .into(),
        )
    }
    if have_any {
        Some(filter)
    } else {
        None
    }
}

pub fn random_vector(dim: usize) -> Vec<f32> {
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
