use crate::args::Args;
use core::option::Option;
use core::option::Option::{None, Some};
use futures::Stream;
use qdrant_client::client::Payload;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    FieldCondition, Filter, GeoPoint, GeoRadius, Match, Range, RepeatedStrings, Vector,
};
use qdrant_client::{Qdrant, QdrantError};
use rand::distr::Distribution;
use rand::prelude::SliceRandom;
use rand::seq::IndexedRandom;
use rand::{rng, Rng};
use serde_json::json;
use std::time::Duration;
use tokio::time::interval;
use tokio_stream::wrappers::IntervalStream;
use tokio_stream::StreamExt;
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
    pub delay_millis: f64, // milliseconds
    pub value: f64,
}

pub fn random_text(num_words: usize, vocab_size: usize) -> String {
    let mut rng = rand::rng();

    let zipf = rand_distr::Zipf::new(f64::from(vocab_size as u32), 1.03).unwrap();

    let zipf_distributed_words: Vec<usize> =
        (0..num_words).map(|_| zipf.sample(&mut rng) as usize).collect();

    let words: Vec<_> = zipf_distributed_words
        .iter()
        .map(|&word_id| format!("word_{}", word_id))
        .collect();

    words.join(" ")
}

pub fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::rng();
    let variant = rng.random_range(0..num_variants);
    format!("keyword_{}", variant)
}

fn random_geo_point() -> GeoPoint {
    let mut rng = rand::rng();
    GeoPoint {
        lat: BERLIN.lat + rng.random_range(-GEO_RADIUS_DEG..=GEO_RADIUS_DEG),
        lon: BERLIN.lon + rng.random_range(-GEO_RADIUS_DEG..=GEO_RADIUS_DEG),
    }
}

pub fn random_payload(args: &Args) -> Payload {
    let mut payload = Payload::new();

    for (idx, variants) in args.keywords.iter().enumerate() {
        let prefix = payload_prefixes(idx);
        payload.insert(
            format!("{}{}", prefix, KEYWORD_PAYLOAD_KEY),
            random_keyword(*variants),
        );
    }

    for (idx, _) in args.float_payloads.iter().enumerate() {
        let prefix = payload_prefixes(idx);
        payload.insert(
            format!("{}{}", prefix, FLOAT_PAYLOAD_KEY),
            rand::rng().random_range(-1.0..1.0),
        );
    }

    for (idx, integer_range) in args.int_payloads.iter().enumerate() {
        let prefix = payload_prefixes(idx);
        let value = rand::rng().random_range(0..*integer_range);
        payload.insert(format!("{}{}", prefix, INTEGERS_PAYLOAD_KEY), value as i64);
    }

    if args.uuid_payloads {
        let value = Uuid::new_v4();
        payload.insert(UUID_PAYLOAD_KEY, value.to_string());
    }

    if args.timestamp_payload {
        payload.insert("timestamp", chrono::Utc::now().to_rfc3339());
    }

    if args.geo_payloads {
        let point = random_geo_point();

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
        payload.insert(
            BOOL_PAYLOAD_KEY,
            rand::rng().random_bool(BOOL_TRUE_RATIO),
        );
    }

    if args.text_payloads {
        let text = random_text(
            args.text_payload_length.unwrap_or(16),
            args.text_payload_vocabulary.unwrap_or(DEFAULT_VOCAB_SIZE),
        );

        payload.insert(TEXT_PAYLOAD_KEY, text);
    }

    payload
}

#[allow(clippy::too_many_arguments)]
pub fn random_filter(
    keywords: Option<usize>,
    float_payloads: bool,
    integer_payload: Option<usize>,
    uuids: &[String],
    match_any: Option<usize>,
    geo_payloads: bool,
    bool_payloads: bool,
    text_payloads_vocab: Option<usize>,
) -> Option<Filter> {
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
        min_should: None,
    };
    let mut have_any = false;
    if let Some(keyword_variants) = keywords {
        let condition = if let Some(match_any) = match_any {
            MatchValue::Keywords(RepeatedStrings {
                strings: (0..match_any)
                    .map(|_| random_keyword(keyword_variants))
                    .collect(),
            })
        } else {
            MatchValue::Keyword(random_keyword(keyword_variants))
        };

        have_any = true;
        filter.must.push(
            FieldCondition {
                key: KEYWORD_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(condition),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                geo_polygon: None,
                values_count: None,
                datetime_range: None,
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
                datetime_range: None,
            }
            .into(),
        )
    }
    if let Some(integer_range) = integer_payload {
        have_any = true;
        let rand_int = rand::rng().random_range(0..integer_range);
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
                datetime_range: None,
            }
            .into(),
        )
    }

    if !uuids.is_empty() {
        have_any = true;
        let mut rng = rng();
        let random = uuids.choose(&mut rng).unwrap();
        filter.must.push(
            FieldCondition {
                key: UUID_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Keyword(random.to_string())),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                geo_polygon: None,
                values_count: None,
                datetime_range: None,
            }
            .into(),
        )
    }

    if geo_payloads {
        have_any = true;
        let radius = rand::rng().random_range(GEO_RADIUS_METERS_MIN..GEO_RADIUS_METERS_MAX);
        filter.must.push(
            FieldCondition {
                key: GEO_PAYLOAD_KEY.to_string(),
                r#match: None,
                range: None,
                geo_bounding_box: None,
                geo_radius: Some(GeoRadius {
                    center: Some(random_geo_point()),
                    radius: radius as f32,
                }),

                values_count: None,
                geo_polygon: None,
                datetime_range: None,
            }
            .into(),
        )
    }

    if bool_payloads {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: BOOL_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Boolean(
                        rand::rng().random_bool(BOOL_TRUE_RATIO),
                    )),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                values_count: None,
                geo_polygon: None,
                datetime_range: None,
            }
            .into(),
        )
    }

    if let Some(vocab_size) = text_payloads_vocab {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: TEXT_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Text(random_text(2, vocab_size))),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                values_count: None,
                geo_polygon: None,
                datetime_range: None,
            }
            .into(),
        )
    }

    have_any.then_some(filter)
}

pub fn random_vector(args: &Args) -> Vector {
    random_dense_vector(args.dim).into()
}

/// Generate random sparse vector with random size and random values.
/// - `max_size` - maximum size of vector
/// - `sparsity` - how many non-zero values should be in vector
pub fn random_sparse_vector(max_size: usize, sparsity: f64) -> Vec<(u32, f32)> {
    let mut rng = rand::rng();
    let size = rng.random_range(1..max_size);
    // (index, value)
    let mut pairs = Vec::with_capacity(size);
    for i in 1..=size {
        // probability of skipping a dimension to make the vectors sparse
        let skip = !rng.random_bool(sparsity);
        if skip {
            continue;
        }
        // Only positive values are generated to make sure to hit the pruning path.
        pairs.push((i as u32, rng.random_range(0.0..10.0) as f32));
    }
    pairs
}

pub fn random_dense_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

pub fn random_vector_name(max: usize) -> String {
    let mut rng = rand::rng();
    format!("{}", rng.random_range(0..max))
}

pub async fn retry_with_clients<'a, R, T: std::future::Future<Output = Result<R, QdrantError>>>(
    clients: &'a [Qdrant],
    args: &Args,
    mut call: impl FnMut(&'a Qdrant) -> T,
) -> anyhow::Result<R> {
    let mut permutation = (0..clients.len()).collect::<Vec<_>>();
    let mut res = Err(anyhow::anyhow!("No clients"));

    for attempt in 0..=args.retries {
        permutation.shuffle(&mut rand::rng());
        for client_id in &permutation {
            let client = clients.get(*client_id).unwrap();

            res = call(client).await.map_err(|i| i.into());

            if res.is_ok() {
                return res;
            }
        }

        let is_last = attempt >= args.retries;
        if !is_last {
            if let Err(err) = &res {
                warn!("Request failed at attempt {}: {err}", attempt + 1);
            }

            tokio::time::sleep(Duration::from_secs_f32(args.retry_interval.max(0.0))).await;
        }
    }

    res
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
        format!("payload_{}_", id)
    }
}
