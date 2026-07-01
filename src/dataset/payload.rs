use serde_json::Value;

/// Pass-through for dataset JSON values into Qdrant payload fields.
pub fn json_to_payload_value(value: Value) -> Value {
    value
}
