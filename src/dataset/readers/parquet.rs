//! Parquet payload and (optional) vector reading.
//!
//! By default a parquet part is payloads only, paired with a separate vector
//! source (typically a `.npy` of the same row count) — that is the LAION shape.
//! FineWeb-style corpora put dense and/or sparse embeddings in the same file;
//! set `vector_column` / `sparse_column` on the dataset config to read those.
//!
//! Access is a *streaming cursor* rather than a decode-the-whole-file cache.
//! Upload walks point ids in order, so the reader keeps one live row iterator
//! plus a small ring of recently decoded rows to absorb the jitter of several
//! parallel workers sitting a few batches apart. That keeps resident memory
//! flat regardless of file size — decoding a whole LAION metadata part up front
//! would cost hundreds of MB and stall the pipeline before the first batch.
//!
//! A read *behind* the ring (random-order ids, e.g. `--max-id`) rewinds, but
//! only to the start of the containing row group, so the worst case is one row
//! group of re-decoding rather than the whole file.

use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::reader::{ChunkReader, FileReader, Length, SerializedFileReader};
use parquet::file::serialized_reader::ReadOptionsBuilder;
use parquet::record::reader::RowIter;
use parquet::record::{Field, List, Map, Row};
use parquet::schema::types::Type as SchemaType;
use serde_json::{Map as JsonMap, Number, Value};

/// Rows kept decoded behind the cursor. Sized to comfortably cover
/// `--parallel` × `--batch-size` workers straddling a boundary.
const RING_ROWS: usize = 16 * 1024;

pub struct ParquetReader {
    path: PathBuf,
    num_rows: usize,
    /// Global index of each row group's first row, plus a final total.
    group_starts: Vec<usize>,
    /// Projected column names, in file order. `None` ⇒ every column.
    columns: Option<Vec<String>>,
    /// Value substituted for nulls and non-finite floats. `None` ⇒ omit the field.
    fill_null: Option<Value>,
    /// Column holding a dense vector (list of floats), if any.
    vector_column: Option<String>,
    /// Column holding a sparse vector struct, if any.
    sparse_column: Option<String>,
    cursor: Mutex<Cursor>,
}

#[derive(Default)]
struct Cursor {
    iter: Option<RowIter<'static>>,
    /// Global index of the row `iter` will yield next.
    next: usize,
    ring: VecDeque<(usize, Value)>,
}

impl ParquetReader {
    /// Open `path`, keeping `columns` (default: all) minus `exclude`.
    ///
    /// When `vector_column` / `sparse_column` are set and `columns` is omitted,
    /// the projection collapses to just those vector columns so payload text is
    /// not decoded on a vectors-only read.
    pub fn open(
        path: &Path,
        columns: Option<&[String]>,
        exclude: &[String],
        fill_null: Option<&Value>,
        vector_column: Option<&str>,
        sparse_column: Option<&str>,
    ) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = SerializedFileReader::new(file)
            .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
        let metadata = reader.metadata();

        let num_rows = metadata.file_metadata().num_rows().max(0) as usize;
        let mut group_starts = Vec::with_capacity(metadata.num_row_groups() + 1);
        let mut running = 0usize;
        for i in 0..metadata.num_row_groups() {
            group_starts.push(running);
            running += metadata.row_group(i).num_rows().max(0) as usize;
        }
        group_starts.push(running);

        let available: Vec<String> = metadata
            .file_metadata()
            .schema_descr()
            .root_schema()
            .get_fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect();

        for name in vector_column.into_iter().chain(sparse_column) {
            if !available.iter().any(|c| c == name) {
                bail!(
                    "{} has no column {name:?}; available columns: {}",
                    path.display(),
                    available.join(", ")
                );
            }
        }

        // Vectors-only read: if the caller named vector columns and did not
        // ask for a payload projection, decode just those columns.
        let effective_columns: Option<Vec<String>> = if columns.is_none()
            && exclude.is_empty()
            && (vector_column.is_some() || sparse_column.is_some())
        {
            let mut cols = Vec::new();
            if let Some(name) = vector_column {
                cols.push(name.to_string());
            }
            if let Some(name) = sparse_column
                && !cols.iter().any(|c| c == name)
            {
                cols.push(name.to_string());
            }
            Some(cols)
        } else {
            None
        };
        let columns = match effective_columns.as_deref() {
            Some(cols) => resolve_projection(&available, Some(cols), exclude, path)?,
            None => resolve_projection(&available, columns, exclude, path)?,
        };

        Ok(ParquetReader {
            path: path.to_path_buf(),
            num_rows,
            group_starts,
            columns,
            fill_null: fill_null.cloned(),
            vector_column: vector_column.map(str::to_string),
            sparse_column: sparse_column.map(str::to_string),
            cursor: Mutex::new(Cursor::default()),
        })
    }

    pub fn num_points(&self) -> usize {
        self.num_rows
    }

    /// Dense vector from `vector_column` at row `idx`.
    pub fn dense_vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        let column = self.vector_column.as_deref().with_context(|| {
            format!(
                "{}: dense vectors require `vector_column` on the parquet dataset",
                self.path.display()
            )
        })?;
        let row = self
            .payload_object(idx)?
            .with_context(|| format!("{}: row {idx} is past the end", self.path.display()))?;
        let value = row.get(column).with_context(|| {
            format!(
                "{}: row {idx} has no dense vector column {column:?}",
                self.path.display()
            )
        })?;
        json_to_dense(value).with_context(|| {
            format!(
                "{}: row {idx} column {column:?} is not a list of floats",
                self.path.display()
            )
        })
    }

    /// Sparse vector from `sparse_column` at row `idx`.
    pub fn sparse_vector_at(&self, idx: usize) -> Result<Vec<(u32, f32)>> {
        let column = self.sparse_column.as_deref().with_context(|| {
            format!(
                "{}: sparse vectors require `sparse_column` on the parquet dataset",
                self.path.display()
            )
        })?;
        let row = self
            .payload_object(idx)?
            .with_context(|| format!("{}: row {idx} is past the end", self.path.display()))?;
        let value = row.get(column).with_context(|| {
            format!(
                "{}: row {idx} has no sparse vector column {column:?}",
                self.path.display()
            )
        })?;
        json_to_sparse(value).with_context(|| {
            format!(
                "{}: row {idx} column {column:?} is not `{{indices, values}}`",
                self.path.display()
            )
        })
    }

    /// The whole payload object for a row.
    pub fn payload_object(&self, idx: usize) -> Result<Option<Value>> {
        if idx >= self.num_rows {
            return Ok(None);
        }
        let mut cursor = self.cursor.lock().unwrap();

        if let Some((_, value)) = cursor.ring.iter().find(|(i, _)| *i == idx) {
            return Ok(Some(value.clone()));
        }
        // Behind the ring (or nothing decoded yet): restart at the row group
        // holding `idx` so a rewind costs one group, not the whole file.
        if cursor.iter.is_none() || idx < cursor.next {
            let group = self.group_of(idx);
            cursor.iter = Some(self.open_iter(group)?);
            cursor.next = self.group_starts[group];
            cursor.ring.clear();
        }

        while cursor.next <= idx {
            let row = cursor
                .iter
                .as_mut()
                .expect("iterator was just installed")
                .next()
                .with_context(|| {
                    format!(
                        "{} ended at row {} while reading row {idx}",
                        self.path.display(),
                        cursor.next
                    )
                })?
                .with_context(|| format!("failed to decode {} row {idx}", self.path.display()))?;

            let value = self.row_to_value(&row);
            let at = cursor.next;
            cursor.ring.push_back((at, value));
            if cursor.ring.len() > RING_ROWS {
                cursor.ring.pop_front();
            }
            cursor.next += 1;
        }

        Ok(cursor
            .ring
            .back()
            .filter(|(i, _)| *i == idx)
            .map(|(_, value)| value.clone()))
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        Ok(self
            .payload_object(idx)?
            .and_then(|object| object.get(field).cloned()))
    }

    /// Index of the row group containing global row `idx`.
    fn group_of(&self, idx: usize) -> usize {
        match self.group_starts.binary_search(&idx) {
            Ok(exact) => exact.min(self.group_starts.len().saturating_sub(2)),
            Err(next) => next - 1,
        }
    }

    /// A row iterator positioned at the first row of row group `group`.
    fn open_iter(&self, group: usize) -> Result<RowIter<'static>> {
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))?;
        let options = ReadOptionsBuilder::new()
            .with_predicate(Box::new(move |_, i| i >= group))
            .build();
        let reader = SerializedFileReader::new_with_options(file, options)
            .with_context(|| format!("failed to open {}", self.path.display()))?;

        let projection = self
            .columns
            .as_ref()
            .map(|names| project_schema(reader.metadata().file_metadata().schema(), names))
            .transpose()?;

        RowIter::from_file_into(Box::new(reader))
            .project(projection)
            .with_context(|| format!("failed to project columns of {}", self.path.display()))
    }

    fn row_to_value(&self, row: &Row) -> Value {
        let mut object = JsonMap::new();
        for (name, field) in row.get_column_iter() {
            match self.field_to_value(field) {
                Some(value) => {
                    object.insert(name.clone(), value);
                }
                None => {
                    if let Some(fill) = &self.fill_null {
                        object.insert(name.clone(), fill.clone());
                    }
                }
            }
        }
        Value::Object(object)
    }

    /// `None` for values with no JSON representation (null, NaN, ±inf,
    /// non-UTF-8 bytes) — the caller then omits the field or substitutes
    /// `fill_null`.
    fn field_to_value(&self, field: &Field) -> Option<Value> {
        Some(match field {
            Field::Null => return None,
            Field::Bool(v) => Value::Bool(*v),
            Field::Byte(v) => Value::Number((*v).into()),
            Field::Short(v) => Value::Number((*v).into()),
            Field::Int(v) => Value::Number((*v).into()),
            Field::Long(v) => Value::Number((*v).into()),
            Field::UByte(v) => Value::Number((*v).into()),
            Field::UShort(v) => Value::Number((*v).into()),
            Field::UInt(v) => Value::Number((*v).into()),
            Field::ULong(v) => Value::Number((*v).into()),
            Field::Date(v) => Value::Number((*v).into()),
            Field::TimeMillis(v) => Value::Number((*v).into()),
            Field::TimeMicros(v) => Value::Number((*v).into()),
            Field::TimestampMillis(v) => Value::Number((*v).into()),
            Field::TimestampMicros(v) => Value::Number((*v).into()),
            Field::Float16(v) => number(f32::from(*v) as f64)?,
            Field::Float(v) => number(*v as f64)?,
            Field::Double(v) => number(*v)?,
            Field::Decimal(v) => number(decimal_to_f64(v))?,
            Field::Str(v) => Value::String(v.clone()),
            Field::Bytes(v) => Value::String(std::str::from_utf8(v.data()).ok()?.to_string()),
            Field::Group(row) => self.row_to_value(row),
            Field::ListInternal(list) => self.list_to_value(list),
            Field::MapInternal(map) => self.map_to_value(map),
        })
    }

    fn list_to_value(&self, list: &List) -> Value {
        Value::Array(
            list.elements()
                .iter()
                .map(|element| self.field_to_value(element).unwrap_or(Value::Null))
                .collect(),
        )
    }

    fn map_to_value(&self, map: &Map) -> Value {
        let mut object = JsonMap::new();
        for (key, value) in map.entries() {
            // JSON object keys are strings; render the key's scalar form.
            let key = match self.field_to_value(key) {
                Some(Value::String(s)) => s,
                Some(other) => other.to_string(),
                None => continue,
            };
            object.insert(key, self.field_to_value(value).unwrap_or(Value::Null));
        }
        Value::Object(object)
    }
}

/// Row count of a local parquet file, read from its footer.
pub fn parquet_row_count(path: &Path) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
    Ok(reader.metadata().file_metadata().num_rows().max(0) as usize)
}

/// Length of the thrift footer described by the last 8 bytes of a parquet file,
/// *including* those 8 bytes — i.e. the smallest tail that holds the metadata.
///
/// `None` when `tail` is too short to hold even the trailer.
pub fn parquet_footer_len(tail: &[u8]) -> Result<Option<usize>> {
    if tail.len() < FOOTER_TRAILER {
        return Ok(None);
    }
    let trailer = &tail[tail.len() - FOOTER_TRAILER..];
    if &trailer[4..] != PARQUET_MAGIC {
        bail!("not a parquet file (missing PAR1 trailer)");
    }
    let len = u32::from_le_bytes(trailer[..4].try_into().unwrap()) as usize;
    Ok(Some(len + FOOTER_TRAILER))
}

/// Row count read from the *tail* of a parquet file of `total_len` bytes.
///
/// Parquet keeps its metadata at the end, so this sizes a remote part from one
/// ranged request instead of a download. `tail` must reach back at least
/// [`parquet_footer_len`] bytes from the end.
pub fn parquet_row_count_from_tail(tail: &[u8], total_len: u64) -> Result<usize> {
    let needed =
        parquet_footer_len(tail)?.context("tail is too short to hold a parquet trailer")?;
    if needed > tail.len() {
        bail!(
            "parquet footer is {needed} bytes but only {} were fetched",
            tail.len()
        );
    }
    if (tail.len() as u64) > total_len {
        bail!(
            "tail of {} bytes exceeds the file length {total_len}",
            tail.len()
        );
    }

    let reader = TailChunkReader {
        total_len,
        start: total_len - tail.len() as u64,
        tail: Bytes::copy_from_slice(tail),
    };
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&reader)
        .context("failed to decode the parquet footer")?;
    Ok(metadata.file_metadata().num_rows().max(0) as usize)
}

const FOOTER_TRAILER: usize = 8;
const PARQUET_MAGIC: &[u8] = b"PAR1";

/// Presents the last `tail.len()` bytes of a file as if the whole file were
/// available, so the parquet metadata reader — which only ever seeks from the
/// end — can work against a ranged response.
struct TailChunkReader {
    total_len: u64,
    start: u64,
    tail: Bytes,
}

impl TailChunkReader {
    fn slice_from(&self, start: u64) -> parquet::errors::Result<Bytes> {
        if start < self.start || start > self.total_len {
            return Err(ParquetError::General(format!(
                "parquet metadata read at {start} falls outside the fetched tail \
                 [{}, {})",
                self.start, self.total_len
            )));
        }
        Ok(self.tail.slice((start - self.start) as usize..))
    }
}

impl Length for TailChunkReader {
    fn len(&self) -> u64 {
        self.total_len
    }
}

impl ChunkReader for TailChunkReader {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        self.slice_from(start).map(bytes::Buf::reader)
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let slice = self.slice_from(start)?;
        if slice.len() < length {
            return Err(ParquetError::General(format!(
                "parquet metadata read of {length} bytes at {start} runs past the fetched tail"
            )));
        }
        Ok(slice.slice(..length))
    }
}

/// JSON has no NaN or infinity — those become "no value", same as null.
fn number(value: f64) -> Option<Value> {
    Number::from_f64(value).map(Value::Number)
}

/// Decode a JSON array of numbers into an `f32` dense vector.
fn json_to_dense(value: &Value) -> Result<Vec<f32>> {
    let Value::Array(items) = value else {
        bail!("expected a JSON array");
    };
    items
        .iter()
        .map(|item| match item {
            Value::Number(n) => n
                .as_f64()
                .map(|f| f as f32)
                .context("non-finite float in dense vector"),
            _ => bail!("dense vector element is not a number"),
        })
        .collect()
}

/// Decode `{indices: [...], values: [...]}` into sparse index/value pairs.
fn json_to_sparse(value: &Value) -> Result<Vec<(u32, f32)>> {
    let object = value
        .as_object()
        .context("expected a JSON object with indices and values")?;
    let indices = object
        .get("indices")
        .and_then(Value::as_array)
        .context("sparse vector missing `indices` array")?;
    let values = object
        .get("values")
        .and_then(Value::as_array)
        .context("sparse vector missing `values` array")?;
    if indices.len() != values.len() {
        bail!(
            "sparse vector indices ({}) and values ({}) lengths differ",
            indices.len(),
            values.len()
        );
    }
    indices
        .iter()
        .zip(values)
        .map(|(idx, val)| {
            let index = match idx {
                Value::Number(n) => n
                    .as_u64()
                    .context("sparse index is not an unsigned integer")?
                    as u32,
                _ => bail!("sparse index is not a number"),
            };
            let value = match val {
                Value::Number(n) => n
                    .as_f64()
                    .map(|f| f as f32)
                    .context("non-finite float in sparse values")?,
                _ => bail!("sparse value is not a number"),
            };
            Ok((index, value))
        })
        .collect()
}

/// Best-effort decimal → float. Payload filters are numeric, and Qdrant has no
/// fixed-point payload type, so precision beyond f64 has nowhere to go anyway.
fn decimal_to_f64(decimal: &parquet::data_type::Decimal) -> f64 {
    let bytes = decimal.data();
    // Big-endian two's complement, sign-extended into i128.
    let mut unscaled: i128 = if bytes.first().is_some_and(|b| b & 0x80 != 0) {
        -1
    } else {
        0
    };
    for byte in bytes {
        unscaled = (unscaled << 8) | i128::from(*byte);
    }
    unscaled as f64 / 10f64.powi(decimal.scale())
}

/// Resolve `columns` / `exclude` against the file's actual columns.
fn resolve_projection(
    available: &[String],
    columns: Option<&[String]>,
    exclude: &[String],
    path: &Path,
) -> Result<Option<Vec<String>>> {
    for name in columns.unwrap_or_default().iter().chain(exclude) {
        if !available.contains(name) {
            bail!(
                "{} has no column {name:?}; available columns: {}",
                path.display(),
                available.join(", ")
            );
        }
    }

    if columns.is_none() && exclude.is_empty() {
        return Ok(None);
    }
    let kept: Vec<String> = available
        .iter()
        .filter(|name| columns.is_none_or(|wanted| wanted.contains(name)))
        .filter(|name| !exclude.contains(name))
        .cloned()
        .collect();
    if kept.is_empty() {
        bail!("{}: `columns`/`exclude` select no columns", path.display());
    }
    Ok(Some(kept))
}

/// Rebuild the root group type with only `names`, preserving file order.
fn project_schema(schema: &SchemaType, names: &[String]) -> Result<SchemaType> {
    let fields = schema
        .get_fields()
        .iter()
        .filter(|field| names.iter().any(|name| name == field.name()))
        .cloned()
        .collect();
    SchemaType::group_type_builder(schema.name())
        .with_fields(fields)
        .build()
        .context("failed to build parquet projection")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::fixtures::write_parquet;

    fn open(path: &Path) -> ParquetReader {
        ParquetReader::open(path, None, &[], None, None, None).unwrap()
    }

    #[test]
    fn reads_rows_in_order_across_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 250, 64);

        let reader = open(&path);
        assert_eq!(reader.num_points(), 250);
        for i in 0..250 {
            let row = reader.payload_object(i).unwrap().unwrap();
            assert_eq!(row["id"], i as i64);
            assert_eq!(row["url"], format!("http://example.com/{i}"));
            assert_eq!(row["nsfw"], i % 2 == 0);
        }
    }

    /// NaN and null have no JSON form; by default the field is simply absent,
    /// which Qdrant treats as "no value" rather than a bogus 0.
    #[test]
    fn omits_nan_and_null_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let reader = open(&path);
        let hole = reader.payload_object(3).unwrap().unwrap();
        assert!(!hole.as_object().unwrap().contains_key("similarity"));
        assert!(!hole.as_object().unwrap().contains_key("caption"));

        let full = reader.payload_object(4).unwrap().unwrap();
        assert_eq!(full["similarity"], 4.0);
        assert_eq!(full["caption"], "caption 4");
    }

    #[test]
    fn fill_null_substitutes_a_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let reader = ParquetReader::open(&path, None, &[], Some(&Value::from(0)), None, None).unwrap();
        let hole = reader.payload_object(3).unwrap().unwrap();
        assert_eq!(hole["similarity"], 0);
        assert_eq!(hole["caption"], 0);
    }

    #[test]
    fn exclude_drops_a_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let reader =
            ParquetReader::open(&path, None, std::slice::from_ref(&"url".to_string()), None, None, None)
                .unwrap();
        let row = reader.payload_object(1).unwrap().unwrap();
        assert!(!row.as_object().unwrap().contains_key("url"));
        assert_eq!(row["id"], 1);
    }

    #[test]
    fn columns_keeps_only_the_listed_ones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let wanted = vec!["id".to_string(), "similarity".to_string()];
        let reader = ParquetReader::open(&path, Some(&wanted), &[], None, None, None).unwrap();
        let row = reader.payload_object(1).unwrap().unwrap();
        let keys: Vec<&String> = row.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["id", "similarity"]);
    }

    #[test]
    fn unknown_column_names_are_rejected_with_the_available_ones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let err = ParquetReader::open(&path, None, &["exif".to_string()], None, None, None)
            .map(|_| ())
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("exif"), "{message}");
        assert!(message.contains("similarity"), "{message}");
    }

    /// Backward seeks must still return the right row (they rewind to the
    /// containing row group), and forward jumps must not skip rows.
    #[test]
    fn serves_out_of_order_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 500, 64);

        let reader = open(&path);
        for i in [499usize, 0, 250, 251, 12, 499, 13] {
            assert_eq!(reader.payload_object(i).unwrap().unwrap()["id"], i as i64);
        }
    }

    #[test]
    fn reads_a_single_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        let reader = open(&path);
        assert_eq!(
            reader.payload_field(4, "caption").unwrap().unwrap(),
            "caption 4"
        );
        assert_eq!(reader.payload_field(3, "caption").unwrap(), None);
    }

    #[test]
    fn out_of_range_row_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.parquet");
        write_parquet(&path, 0, 10, 64);

        assert_eq!(open(&path).payload_object(10).unwrap(), None);
    }

    fn fineweb_tiny() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dataset/testdata/fineweb_tiny.parquet")
    }

    #[test]
    fn reads_dense_vectors_from_vector_column() {
        let path = fineweb_tiny();
        let reader =
            ParquetReader::open(&path, None, &[], None, Some("dense_embedding"), None).unwrap();
        assert_eq!(reader.num_points(), 5);
        let v = reader.dense_vector_at(0).unwrap();
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[7], 7.0);
        let v2 = reader.dense_vector_at(2).unwrap();
        assert_eq!(v2[0], 2.0);
        assert_eq!(v2[1], 3.0);
    }

    #[test]
    fn reads_sparse_vectors_from_sparse_column() {
        let path = fineweb_tiny();
        let reader =
            ParquetReader::open(&path, None, &[], None, None, Some("sparse_embedding")).unwrap();
        let pairs = reader.sparse_vector_at(0).unwrap();
        assert_eq!(pairs, vec![(1, 0.1), (3, 0.2), (5, 0.3)]);
        assert_eq!(reader.sparse_vector_at(2).unwrap(), vec![(7, 1.0)]);
    }

    #[test]
    fn dense_vector_requires_vector_column() {
        let path = fineweb_tiny();
        let reader = ParquetReader::open(&path, None, &[], None, None, None).unwrap();
        let err = reader.dense_vector_at(0).unwrap_err().to_string();
        assert!(err.contains("vector_column"), "{err}");
    }

    #[test]
    fn exclude_keeps_payload_without_embeddings() {
        let path = fineweb_tiny();
        let exclude = vec![
            "dense_embedding".to_string(),
            "sparse_embedding".to_string(),
            "text".to_string(),
        ];
        let reader = ParquetReader::open(&path, None, &exclude, None, None, None).unwrap();
        let row = reader.payload_object(1).unwrap().unwrap();
        let obj = row.as_object().unwrap();
        assert!(!obj.contains_key("dense_embedding"));
        assert!(!obj.contains_key("text"));
        assert_eq!(obj["token_count"], 101);
        assert_eq!(obj["dump"], "CC-MAIN-2016-01");
    }

    /// Optional smoke against a real FineWeb shard (set `FINEWEB_SAMPLE` to the
    /// path of a `*.parquet` from Qdrant/FineWeb-10B).
    #[test]
    fn reads_a_real_fineweb_sample_when_configured() {
        let Ok(path) = std::env::var("FINEWEB_SAMPLE") else {
            return;
        };
        let path = Path::new(&path);
        if !path.exists() {
            return;
        }
        let dense = ParquetReader::open(path, None, &[], None, Some("dense_embedding"), None)
            .unwrap();
        let v = dense.dense_vector_at(0).unwrap();
        assert_eq!(v.len(), 768, "mGTE dense vectors are 768-d");

        let sparse =
            ParquetReader::open(path, None, &[], None, None, Some("sparse_embedding")).unwrap();
        let pairs = sparse.sparse_vector_at(0).unwrap();
        assert!(!pairs.is_empty());
        assert!(pairs.windows(2).all(|w| w[0].0 < w[1].0) || pairs.len() == 1);
    }
}
