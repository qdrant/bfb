use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use crate::common::Timing;


pub fn save_timings_as_jsonl(jsonl_path: &str, absolute_time: bool, timings: &[Timing], start_timestamp_millis: f64, value_name: &str) -> std::io::Result<()> {
    let mut data = Vec::new();
    for timing in timings.iter() {
        let mut record = HashMap::new();
        if absolute_time {
            record.insert(
                "timestamp".to_string(),
                start_timestamp_millis + timing.delay_millis,
            );
        } else {
            record.insert("delay".to_string(), timing.delay_millis);
        }
        record.insert(value_name.to_string(), timing.value);
        data.push(record);
    }

    save_data_as_jsonl(jsonl_path, &data)
}

pub fn save_data_as_jsonl(path: &str, data: &[HashMap<String, f64>]) -> std::io::Result<()> {
    if path == "-" {
        for record in data {
            let json = serde_json::to_string(&record)?;
            println!("{}", json);
        }
        return Ok(());
    }

    let path = std::path::Path::new(path);

    let mut file = File::options().write(true).create(true).truncate(true).open(path)?;

    for record in data {
        let json = serde_json::to_string(&record)?;
        writeln!(&mut file, "{}", json)?;
    }

    Ok(())
}
