//! Per-query filter conditions from a dataset's query set.
//!
//! `tests.jsonl` in the ann-filtering-benchmark-datasets layout pairs every
//! query with the conditions it was answered under:
//!
//! ```json
//! {"query": [..], "conditions": {"and": [{"similarity": {"range": {"gt": 0.34}}}]},
//!  "closest_ids": [..], "closest_scores": [..]}
//! ```
//!
//! The `closest_ids` are the ground truth **for the filtered query**, so a
//! benchmark that ignores the conditions scores an unfiltered search against
//! the answers to a different question. On `laion-small-clip` — half of whose
//! queries carry a `range` condition — that reads as recall 0.64 instead of
//! 0.99.
//!
//! The dialect is the one vector-db-benchmark defines in
//! `engine/base_client/parser.py`, so numbers stay comparable with its results:
//!
//! ```text
//! conditions := { "and": [entry, ..], "or": [entry, ..] }   both optional
//! entry      := { field_name: { condition_type: criteria } }
//! condition  := "match" { "value": string|number }
//!             | "range" { "lt"?, "gt"?, "lte"?, "gte"? }
//!             | "geo"   { "lat", "lon", "radius" }
//! ```
//!
//! `and` maps to `must`, `or` to `should`.

use anyhow::{Context, Result, bail};
use qdrant_client::qdrant::{Condition, Filter, GeoPoint, GeoRadius, Range};
use serde_json::Value;

/// Parse a `conditions` object into a filter.
///
/// Returns `None` for the query sets that have no conditions at all — `null`,
/// or the `{}` that the `-no-filters` datasets write on every row — so an
/// unfiltered dataset costs nothing and stays byte-identical to before.
pub fn parse(conditions: Option<&Value>) -> Result<Option<Filter>> {
    let Some(value) = conditions else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value.as_object().context("`conditions` is not an object")?;
    if object.is_empty() {
        return Ok(None);
    }

    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
        min_should: None,
    };
    for (operator, entries) in object {
        let target = match operator.as_str() {
            "and" => &mut filter.must,
            "or" => &mut filter.should,
            other => bail!("unknown `conditions` operator {other:?}; expected \"and\" or \"or\""),
        };
        let entries = entries
            .as_array()
            .with_context(|| format!("`conditions.{operator}` is not an array"))?;
        for entry in entries {
            push_entry(entry, target)?;
        }
    }

    // `{"and": []}` carries no conditions; an empty filter would still be sent
    // as a filtered search, which is not the same request.
    if filter.must.is_empty() && filter.should.is_empty() {
        return Ok(None);
    }
    Ok(Some(filter))
}

/// One `{field: {condition_type: criteria}}` entry. A single entry may name
/// several fields, and a field several condition types; every one becomes its
/// own condition, as in vector-db-benchmark's `create_condition_subfilters`.
fn push_entry(entry: &Value, out: &mut Vec<Condition>) -> Result<()> {
    let fields = entry
        .as_object()
        .context("`conditions` entry is not an object")?;
    for (field, by_type) in fields {
        let by_type = by_type
            .as_object()
            .with_context(|| format!("`conditions` entry for {field:?} is not an object"))?;
        for (condition_type, criteria) in by_type {
            out.push(
                build(field, condition_type, criteria)
                    .with_context(|| format!("`conditions` on field {field:?}"))?,
            );
        }
    }
    Ok(())
}

fn build(field: &str, condition_type: &str, criteria: &Value) -> Result<Condition> {
    match condition_type {
        "match" => {
            let value = criteria.get("value").context("`match` has no `value`")?;
            match value {
                Value::String(text) => Ok(Condition::matches(field, text.clone())),
                Value::Bool(flag) => Ok(Condition::matches(field, *flag)),
                Value::Number(number) => {
                    let int = number
                        .as_i64()
                        .context("`match.value` is a non-integer number")?;
                    Ok(Condition::matches(field, int))
                }
                other => bail!("`match.value` must be a string, integer or bool, got {other}"),
            }
        }
        "range" => {
            let bound = |name: &str| -> Result<Option<f64>> {
                match criteria.get(name) {
                    None | Some(Value::Null) => Ok(None),
                    Some(value) => {
                        Ok(Some(value.as_f64().with_context(|| {
                            format!("`range.{name}` is not a number")
                        })?))
                    }
                }
            };
            let range = Range {
                lt: bound("lt")?,
                gt: bound("gt")?,
                gte: bound("gte")?,
                lte: bound("lte")?,
            };
            if range.lt.is_none()
                && range.gt.is_none()
                && range.gte.is_none()
                && range.lte.is_none()
            {
                bail!("`range` has no bounds");
            }
            Ok(Condition::range(field, range))
        }
        "geo" => {
            let number = |name: &str| -> Result<f64> {
                criteria
                    .get(name)
                    .and_then(Value::as_f64)
                    .with_context(|| format!("`geo` is missing a numeric `{name}`"))
            };
            Ok(Condition::geo_radius(
                field,
                GeoRadius {
                    center: Some(GeoPoint {
                        lat: number("lat")?,
                        lon: number("lon")?,
                    }),
                    radius: number("radius")? as f32,
                },
            ))
        }
        other => {
            bail!("unknown condition type {other:?}; expected \"match\", \"range\" or \"geo\"")
        }
    }
}

#[cfg(test)]
mod tests {
    use qdrant_client::qdrant::condition::ConditionOneOf;
    use serde_json::json;

    use super::*;

    fn field_conditions(filter: &Filter) -> (usize, usize) {
        (filter.must.len(), filter.should.len())
    }

    /// The `-no-filters` datasets write `{}` on every row, and older query sets
    /// omit the field. Both must stay an unfiltered search, not an empty filter.
    #[test]
    fn absent_and_empty_conditions_are_unfiltered() {
        assert!(parse(None).unwrap().is_none());
        assert!(parse(Some(&Value::Null)).unwrap().is_none());
        assert!(parse(Some(&json!({}))).unwrap().is_none());
        assert!(parse(Some(&json!({"and": []}))).unwrap().is_none());
    }

    /// The exact shape half of laion-small-clip's query set carries.
    #[test]
    fn parses_the_laion_range_condition() {
        let filter = parse(Some(&json!({
            "and": [{"similarity": {"range": {"gt": 0.3491986757069205}}}]
        })))
        .unwrap()
        .expect("a range condition is a filter");
        assert_eq!(field_conditions(&filter), (1, 0));

        let Some(ConditionOneOf::Field(field)) = &filter.must[0].condition_one_of else {
            panic!("expected a field condition, got {:?}", filter.must[0]);
        };
        assert_eq!(field.key, "similarity");
        let range = field.range.as_ref().expect("range");
        assert_eq!(range.gt, Some(0.3491986757069205));
        assert_eq!(range.gte, None);
        assert_eq!(range.lt, None);
        assert_eq!(range.lte, None);
    }

    /// `and` is `must`, `or` is `should` — the mapping vector-db-benchmark uses.
    #[test]
    fn and_is_must_or_is_should() {
        let filter = parse(Some(&json!({
            "and": [{"a": {"match": {"value": "x"}}}],
            "or": [{"b": {"match": {"value": 80}}}, {"b": {"match": {"value": 2}}}]
        })))
        .unwrap()
        .unwrap();
        assert_eq!(field_conditions(&filter), (1, 2));
    }

    #[test]
    fn parses_geo_and_multi_bound_range() {
        let filter = parse(Some(&json!({
            "and": [
                {"loc": {"geo": {"lat": 52.5, "lon": 13.4, "radius": 1000.0}}},
                {"n": {"range": {"gte": 1, "lt": 9}}}
            ]
        })))
        .unwrap()
        .unwrap();
        assert_eq!(field_conditions(&filter), (2, 0));

        let Some(ConditionOneOf::Field(geo)) = &filter.must[0].condition_one_of else {
            panic!("expected a geo field condition");
        };
        let radius = geo.geo_radius.as_ref().expect("geo_radius");
        assert_eq!(radius.radius, 1000.0);
        assert_eq!(radius.center.as_ref().unwrap().lat, 52.5);

        let Some(ConditionOneOf::Field(range)) = &filter.must[1].condition_one_of else {
            panic!("expected a range field condition");
        };
        let range = range.range.as_ref().expect("range");
        assert_eq!((range.gte, range.lt), (Some(1.0), Some(9.0)));
    }

    /// One entry may name several fields, and a field several condition types;
    /// each becomes its own condition, matching vector-db-benchmark.
    #[test]
    fn one_entry_can_carry_several_conditions() {
        let filter = parse(Some(&json!({
            "and": [{"a": {"match": {"value": "x"}}, "b": {"range": {"gt": 1}}}]
        })))
        .unwrap()
        .unwrap();
        assert_eq!(field_conditions(&filter), (2, 0));
    }

    /// Silently dropping something unrecognized would score the run against
    /// ground truth for a filter that was never applied — the whole bug this
    /// module exists to fix.
    #[test]
    fn unknown_shapes_are_errors_not_silent_drops() {
        for bad in [
            json!({"nand": [{"a": {"match": {"value": 1}}}]}),
            json!({"and": [{"a": {"regex": {"value": "x"}}}]}),
            json!({"and": [{"a": {"range": {}}}]}),
            json!({"and": [{"a": {"match": {}}}]}),
            json!({"and": [{"a": {"geo": {"lat": 1.0, "lon": 2.0}}}]}),
            json!({"and": "not-an-array"}),
        ] {
            assert!(parse(Some(&bad)).is_err(), "should have failed: {bad}");
        }
    }
}
