use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::client::Payload;
use qdrant_client::qdrant::{FieldCondition, Filter, Match};
use qdrant_client::qdrant::r#match::MatchValue;
use rand::Rng;

pub const KEYWORD_PAYLOAD_KEY: &str = "a";

pub fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::thread_rng();
    let variant = rng.gen_range(0..num_variants);
    format!("keyword_{}", variant)
}

pub fn random_payload(keywords: Option<usize>) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords {
        payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(keyword_variants));
    }
    payload
}

pub fn random_filter(keywords: Option<usize>) -> Option<Filter> {
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

pub fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}
