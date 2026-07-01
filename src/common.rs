use crate::args::Args;
use anyhow::Error;
use core::option::Option;
use core::option::Option::{None, Some};
use futures::Stream;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Filter, GeoPoint, GeoRadius, Range, RepeatedStrings, Vector,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::prelude::SliceRandom;
use rand::seq::IndexedRandom;
use serde_json::json;
use std::time::Duration;
use tokio::time::interval;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::IntervalStream;
use tracing::warn;
use uuid::Uuid;

pub const KEYWORD_PAYLOAD_KEY: &str = "a";
pub const FLOAT_PAYLOAD_KEY: &str = "b";

pub const INTEGERS_PAYLOAD_KEY: &str = "c";
pub const UUID_PAYLOAD_KEY: &str = "d";
pub const BOOL_PAYLOAD_KEY: &str = "e";

pub const GEO_PAYLOAD_KEY: &str = "g";

pub const TEXT_PAYLOAD_KEY: &str = "t";

pub const DEFAULT_VOCAB_SIZE: usize = 8192;

const BERLIN: GeoPoint = GeoPoint {
    lat: 52.52437,
    lon: 13.41053,
};
const GEO_RADIUS_DEG: f64 = 1.0;

const GEO_RADIUS_METERS_MIN: f64 = 1000.0;
const GEO_RADIUS_METERS_MAX: f64 = 50000.0;

const BOOL_TRUE_RATIO: f64 = 0.7;

#[derive(Debug, Clone)]
pub struct Timing {
    pub delay_millis: u32,
    pub value: f32,
}

pub fn create_zipf(vocab_size: usize) -> rand_distr::Zipf<f64> {
    rand_distr::Zipf::new(f64::from(vocab_size as u32), 1.03).unwrap()
}

pub fn random_text(rng: &mut impl Rng, num_words: usize, zipf: &rand_distr::Zipf<f64>) -> String {
    let zipf_distributed_words: Vec<usize> =
        (0..num_words).map(|_| zipf.sample(rng) as usize).collect();

    let words: Vec<_> = zipf_distributed_words
        .iter()
        .map(|&word_id| format!("word_{word_id}"))
        .collect();

    words.join(" ")
}

pub fn random_keyword(
    rng: &mut impl Rng,
    num_variants: usize,
    keyword_len_multiplier: usize,
) -> String {
    let variant = rng.random_range(0..num_variants);
    format!("keyword_{variant}").repeat(keyword_len_multiplier)
}

fn random_geo_point(rng: &mut impl Rng) -> GeoPoint {
    GeoPoint {
        lat: BERLIN.lat + rng.random_range(-GEO_RADIUS_DEG..=GEO_RADIUS_DEG),
        lon: BERLIN.lon + rng.random_range(-GEO_RADIUS_DEG..=GEO_RADIUS_DEG),
    }
}

fn estimated_payload_size(args: &Args) -> usize {
    args.keywords.len()
        + args.float_payloads.len()
        + args.int_payloads.len()
        + args.uuid_payloads as usize
        + args.timestamp_payload as usize
        + args.geo_payloads as usize
        + args.bool_payloads as usize
        + args.text_payloads as usize
}

pub fn random_payload(
    rng: &mut impl Rng,
    args: &Args,
    zipf: Option<&rand_distr::Zipf<f64>>,
) -> Payload {
    let payload_size = estimated_payload_size(args);
    let mut payload = Payload::with_capacity(payload_size);

    for (idx, variants) in args.keywords.iter().enumerate() {
        let payload_name = keyword_payload_name(idx);
        if args.max_keywords == 1 {
            payload.insert(
                payload_name,
                random_keyword(rng, *variants, args.keywords_length_multiplier),
            );
        } else {
            let count = rng.random_range(1..=args.max_keywords);
            let values = (0..count)
                .map(|_| random_keyword(rng, *variants, args.keywords_length_multiplier))
                .collect::<Vec<_>>();
            payload.insert(payload_name, values);
        }
    }

    for (idx, _) in args.float_payloads.iter().enumerate() {
        let value = rng.random_range(-1.0..1.0);
        payload.insert(float_payload_name(idx), value);
    }

    for (idx, integer_range) in args.int_payloads.iter().enumerate() {
        let payload_name = int_payload_name(idx);

        if args.max_int_payloads == 1 {
            let value = rng.random_range(0..*integer_range);
            payload.insert(payload_name, value as i64);
        } else {
            let count = rng.random_range(1..=args.max_int_payloads);
            let values = (0..count)
                .map(|_| rng.random_range(0..*integer_range) as i64)
                .collect::<Vec<_>>();
            payload.insert(payload_name, values);
        }
    }

    if args.uuid_payloads {
        let value = Uuid::new_v4();
        payload.insert(UUID_PAYLOAD_KEY, value.to_string());
    }

    if args.timestamp_payload {
        payload.insert("timestamp", chrono::Utc::now().to_rfc3339());
    }

    if args.geo_payloads {
        let point = random_geo_point(rng);

        payload.insert(
            GEO_PAYLOAD_KEY,
            json!(
            {
                "lon": point.lon,
                "lat": point.lat
            }),
        );
    }

    if args.bool_payloads {
        payload.insert(BOOL_PAYLOAD_KEY, rng.random_bool(BOOL_TRUE_RATIO));
    }

    if let Some(zipf) = zipf {
        let text = random_text(rng, args.text_payload_length.unwrap_or(16), zipf);

        payload.insert(TEXT_PAYLOAD_KEY, text);
    }

    payload
}

#[allow(clippy::too_many_arguments)]
pub fn random_filter(
    rng: &mut impl Rng,
    keywords: &[usize],
    float_payloads: &[bool],
    integer_payload: &[usize],
    uuids: &[String],
    match_any: Option<usize>,
    geo_payloads: bool,
    bool_payloads: bool,
    keyword_len_multiplier: usize,
    zipf: Option<&rand_distr::Zipf<f64>>,
) -> Option<Filter> {
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
        min_should: None,
    };
    let mut have_any = false;

    if !keywords.is_empty() {
        // Pick a random keyword payload filter.
        let keyword_index_pos = rng.random_range(0..keywords.len());
        let keyword_variants = keywords[keyword_index_pos];

        let condition = if let Some(match_any) = match_any {
            let strings = (0..match_any)
                .map(|_| random_keyword(rng, keyword_variants, keyword_len_multiplier))
                .collect();
            MatchValue::Keywords(RepeatedStrings { strings })
        } else {
            MatchValue::Keyword(random_keyword(
                rng,
                keyword_variants,
                keyword_len_multiplier,
            ))
        };

        have_any = true;
        filter.must.push(Condition::matches(
            keyword_payload_name(keyword_index_pos),
            condition,
        ));
    }

    if !float_payloads.is_empty() {
        have_any = true;

        let integer_index_pos = rng.random_range(0..float_payloads.len());
        filter.must.push(Condition::range(
            float_payload_name(integer_index_pos),
            Range {
                gt: None,
                gte: Some(0.0),
                lt: None,
                lte: None,
            },
        ))
    }

    if !integer_payload.is_empty() {
        have_any = true;

        let integer_index_pos = rng.random_range(0..integer_payload.len());
        let int_range = integer_payload[integer_index_pos];

        let rand_int = rng.random_range(0..int_range);
        filter.must.push(Condition::range(
            int_payload_name(integer_index_pos),
            Range {
                gt: None,
                gte: Some(rand_int as f64),
                lt: None,
                lte: None,
            },
        ))
    }

    if !uuids.is_empty() {
        have_any = true;
        let random = uuids.choose(rng).unwrap();
        filter
            .must
            .push(Condition::matches(UUID_PAYLOAD_KEY, random.to_string()))
    }

    if geo_payloads {
        have_any = true;
        let radius = rng.random_range(GEO_RADIUS_METERS_MIN..GEO_RADIUS_METERS_MAX);
        filter.must.push(Condition::geo_radius(
            GEO_PAYLOAD_KEY,
            GeoRadius {
                center: Some(random_geo_point(rng)),
                radius: radius as f32,
            },
        ))
    }

    if bool_payloads {
        have_any = true;
        filter.must.push(Condition::matches(
            BOOL_PAYLOAD_KEY,
            rng.random_bool(BOOL_TRUE_RATIO),
        ))
    }

    if let Some(zipf) = zipf {
        have_any = true;
        filter.must.push(Condition::matches_text(
            TEXT_PAYLOAD_KEY,
            random_text(rng, 2, zipf),
        ))
    }

    have_any.then_some(filter)
}

pub fn random_vector(rng: &mut impl Rng, args: &Args) -> Vector {
    let is_uint = args.is_uint8_datatype();
    if let Some(count) = args.multivector_size {
        let multivector: Vec<_> = (0..count)
            .map(|_| random_dense_vector(rng, args.dim, is_uint))
            .collect();
        Vector::new_multi(multivector)
    } else {
        random_dense_vector(rng, args.dim, is_uint).into()
    }
}

/// Generate a random sparse vector with `length` non-zero entries.
/// Indices are drawn uniformly from `1..=vocab_size` (inclusive).
pub fn random_sparse_vector(
    rng: &mut impl Rng,
    vocab_size: usize,
    length: usize,
) -> Vec<(u32, f32)> {
    let vocab_size = vocab_size.max(1);
    let length = length.min(vocab_size);
    if length == 0 {
        return Vec::new();
    }

    let mut seen = std::collections::HashSet::new();
    let mut pairs = Vec::with_capacity(length);
    let mut attempts = 0;
    while pairs.len() < length && attempts < length * 8 {
        attempts += 1;
        let idx = rng.random_range(1..=vocab_size) as u32;
        if seen.insert(idx) {
            // Only positive values are generated to make sure to hit the pruning path.
            pairs.push((idx, rng.random_range(0.0..10.0) as f32));
        }
    }
    pairs
}

pub fn random_dense_vector(rng: &mut impl Rng, dim: usize, is_uint: bool) -> Vec<f32> {
    if is_uint {
        (0..dim).map(|_| rng.random_range(0..255) as f32).collect()
    } else {
        (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
    }
}

pub fn random_vector_name(rng: &mut impl Rng, max: usize) -> String {
    format!("{}", rng.random_range(0..max))
}

pub async fn retry_with_clients<'a, R, T: std::future::Future<Output = Result<R, QdrantError>>>(
    clients: &'a [Qdrant],
    args: &Args,
    mut call: impl FnMut(&'a Qdrant) -> T,
) -> anyhow::Result<R> {
    let mut rng = rand::rng();
    let mut permutation = (0..clients.len()).collect::<Vec<_>>();
    let mut previous_err: Option<Error> = None;

    for attempt in 0..=args.retries {
        permutation.shuffle(&mut rng);
        for client_id in &permutation {
            let client = clients.get(*client_id).unwrap();

            let resp = call(client).await.map_err(|i| i.into());

            match resp {
                Ok(_) => {
                    return resp;
                }
                Err(err) => {
                    previous_err = Some(err);
                }
            }
        }

        let is_last = attempt >= args.retries;
        if !is_last {
            if let Some(err) = &previous_err {
                warn!("Request failed at attempt {}: {err}", attempt + 1);
            }

            tokio::time::sleep(Duration::from_secs_f32(args.retry_interval.max(0.0))).await;
        }
    }

    Err(previous_err.unwrap_or_else(|| anyhow::anyhow!("No clients")))
}

/// Build a stream that will emit a unit value at the given frequency
///
/// If `None` - the stream will emit a unit value every time it is polled.
pub(crate) fn throttler(hz: Option<f32>) -> Box<dyn Stream<Item = ()> + Unpin> {
    match hz
        // Do not support zero or infinite
        .filter(|throttle| *throttle != 0.0 && !throttle.is_nan() && !throttle.is_infinite())
        .map(|throttle| Duration::from_secs_f32(1.0 / throttle))
        // Do not support durations of zero
        .filter(|duration| !duration.is_zero())
    {
        Some(duration) => {
            let interval = interval(duration);
            Box::new(IntervalStream::new(interval).map(|_| ()))
        }
        None => Box::new(futures::stream::repeat(())),
    }
}

pub fn payload_prefixes(id: usize) -> String {
    if id == 0 {
        "".to_string()
    } else {
        format!("payload_{id}_")
    }
}

pub fn keyword_payload_name(id: usize) -> String {
    format!("{}{}", payload_prefixes(id), KEYWORD_PAYLOAD_KEY)
}

pub fn int_payload_name(id: usize) -> String {
    format!("{}{}", payload_prefixes(id), INTEGERS_PAYLOAD_KEY)
}

pub fn float_payload_name(id: usize) -> String {
    format!("{}{}", payload_prefixes(id), FLOAT_PAYLOAD_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_payload_prefixes() {
        assert_eq!(payload_prefixes(0), "");
        assert_eq!(payload_prefixes(1), "payload_1_");
        assert_eq!(payload_prefixes(5), "payload_5_");
    }

    #[test]
    fn test_keyword_payload_name() {
        assert_eq!(keyword_payload_name(0), KEYWORD_PAYLOAD_KEY);
        assert_eq!(
            keyword_payload_name(1),
            format!("payload_1_{KEYWORD_PAYLOAD_KEY}")
        );
    }

    #[test]
    fn test_int_payload_name() {
        assert_eq!(int_payload_name(0), INTEGERS_PAYLOAD_KEY);
        assert_eq!(
            int_payload_name(2),
            format!("payload_2_{INTEGERS_PAYLOAD_KEY}")
        );
    }

    #[test]
    fn test_float_payload_name() {
        assert_eq!(float_payload_name(0), FLOAT_PAYLOAD_KEY);
        assert_eq!(
            float_payload_name(3),
            format!("payload_3_{FLOAT_PAYLOAD_KEY}")
        );
    }

    #[test]
    fn test_random_keyword_format() {
        let mut rng = seeded_rng();
        let kw = random_keyword(&mut rng, 10, 1);
        assert!(kw.starts_with("keyword_"));
    }

    #[test]
    fn test_random_keyword_length_multiplier() {
        let mut rng = seeded_rng();
        let single = random_keyword(&mut rng, 10, 1);
        let mut rng = seeded_rng();
        let triple = random_keyword(&mut rng, 10, 3);
        assert_eq!(triple.len(), single.len() * 3);
    }

    #[test]
    fn test_random_dense_vector_dimension() {
        let mut rng = seeded_rng();
        let vec = random_dense_vector(&mut rng, 128, false);
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_random_dense_vector_float_range() {
        let mut rng = seeded_rng();
        let vec = random_dense_vector(&mut rng, 1000, false);
        for &v in &vec {
            assert!((-1.0..1.0).contains(&v), "value {v} out of [-1, 1) range");
        }
    }

    #[test]
    fn test_random_dense_vector_uint_range() {
        let mut rng = seeded_rng();
        let vec = random_dense_vector(&mut rng, 1000, true);
        for &v in &vec {
            assert!(
                (0.0..255.0).contains(&v),
                "uint8 value {v} out of [0, 255) range"
            );
            assert_eq!(v, v.floor(), "uint8 value {v} should be integral");
        }
    }

    #[test]
    fn test_random_text_word_count() {
        let mut rng = seeded_rng();
        let zipf = create_zipf(DEFAULT_VOCAB_SIZE);
        let text = random_text(&mut rng, 5, &zipf);
        let words: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(words.len(), 5);
        for word in &words {
            assert!(word.starts_with("word_"));
        }
    }

    #[test]
    fn test_random_sparse_vector_bounds() {
        let mut rng = seeded_rng();
        let pairs = random_sparse_vector(&mut rng, 100, 20);
        assert_eq!(pairs.len(), 20);
        for &(idx, val) in &pairs {
            assert!(idx >= 1, "sparse index should be >= 1, got {idx}");
            assert!(
                idx <= 100,
                "sparse index should be <= vocab_size, got {idx}"
            );
            assert!(val >= 0.0, "sparse value should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_random_sparse_vector_empty_with_zero_length() {
        let mut rng = seeded_rng();
        let pairs = random_sparse_vector(&mut rng, 100, 0);
        assert!(pairs.is_empty());
    }
}
