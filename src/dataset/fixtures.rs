//! Dataset files built on the fly for tests: real `.npy` and `.parquet` bytes,
//! so the readers are exercised against the formats rather than against mocks.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

/// Build a minimal little-endian `.npy` v1.0 buffer.
pub fn make_npy(descr: &str, rows: usize, cols: usize, data: &[u8]) -> Vec<u8> {
    let dict =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut header = dict.into_bytes();
    // Pad with spaces so the total (magic + version + len field + header) is a
    // multiple of 64, and terminate with a newline.
    let unpadded = 10 + header.len() + 1;
    header.extend(std::iter::repeat_n(b' ', (64 - unpadded % 64) % 64));
    header.push(b'\n');

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]); // version 1.0
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    out
}

/// A `rows` x `cols` f32 `.npy` whose value at (r, c) is `base + r * cols + c`.
pub fn make_ramp_npy(base: usize, rows: usize, cols: usize) -> Vec<u8> {
    let bytes: Vec<u8> = (0..rows * cols)
        .flat_map(|x| ((base + x) as f32).to_le_bytes())
        .collect();
    make_npy("<f4", rows, cols, &bytes)
}

const MESSAGE: &str = "
    message laionish {
        REQUIRED INT64 id;
        REQUIRED DOUBLE similarity;
        REQUIRED BYTE_ARRAY url (UTF8);
        REQUIRED BOOLEAN nsfw;
        OPTIONAL BYTE_ARRAY caption (UTF8);
    }
";

/// Write `rows` payload rows in groups of `group_size`. Row `i` gets
/// `id = base + i`; every third row has a NaN `similarity` and a null
/// `caption`, mirroring the holes in real LAION metadata.
pub fn write_parquet(path: &Path, base: i64, rows: usize, group_size: usize) {
    let schema = Arc::new(parse_message_type(MESSAGE).unwrap());
    let props = Arc::new(
        WriterProperties::builder()
            .set_max_row_group_row_count(Some(group_size))
            .build(),
    );
    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();

    for start in (0..rows).step_by(group_size) {
        let end = (start + group_size).min(rows);
        let span: Vec<usize> = (start..end).collect();
        let mut group = writer.next_row_group().unwrap();

        let ids: Vec<i64> = span.iter().map(|i| base + *i as i64).collect();
        let mut col = group.next_column().unwrap().unwrap();
        col.typed::<Int64Type>()
            .write_batch(&ids, None, None)
            .unwrap();
        col.close().unwrap();

        let sims: Vec<f64> = span
            .iter()
            .map(|i| if i % 3 == 0 { f64::NAN } else { *i as f64 })
            .collect();
        let mut col = group.next_column().unwrap().unwrap();
        col.typed::<DoubleType>()
            .write_batch(&sims, None, None)
            .unwrap();
        col.close().unwrap();

        let urls: Vec<ByteArray> = span
            .iter()
            .map(|i| ByteArray::from(format!("http://example.com/{i}").as_str()))
            .collect();
        let mut col = group.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(&urls, None, None)
            .unwrap();
        col.close().unwrap();

        let flags: Vec<bool> = span.iter().map(|i| i % 2 == 0).collect();
        let mut col = group.next_column().unwrap().unwrap();
        col.typed::<BoolType>()
            .write_batch(&flags, None, None)
            .unwrap();
        col.close().unwrap();

        // Optional column: definition level 0 = null, 1 = present.
        let defs: Vec<i16> = span.iter().map(|i| i16::from(i % 3 != 0)).collect();
        let captions: Vec<ByteArray> = span
            .iter()
            .filter(|i| *i % 3 != 0)
            .map(|i| ByteArray::from(format!("caption {i}").as_str()))
            .collect();
        let mut col = group.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(&captions, Some(&defs), None)
            .unwrap();
        col.close().unwrap();

        group.close().unwrap();
    }
    writer.close().unwrap();
}
