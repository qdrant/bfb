use std::fs::File;
use std::io::Read;

use anyhow::{Context, Result};

pub(super) fn read_i64_array(file: &mut File, out: &mut [i64]) -> Result<()> {
    let mut buf = vec![0u8; out.len() * 8];
    file.read_exact(&mut buf).context("failed to read i64 array")?;
    for (slot, chunk) in out.iter_mut().zip(buf.chunks_exact(8)) {
        *slot = i64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

pub(super) fn read_i32_array(file: &mut File, out: &mut [i32]) -> Result<()> {
    let mut buf = vec![0u8; out.len() * 4];
    file.read_exact(&mut buf).context("failed to read i32 array")?;
    for (slot, chunk) in out.iter_mut().zip(buf.chunks_exact(4)) {
        *slot = i32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

pub(super) fn read_f32_array(file: &mut File, out: &mut [f32]) -> Result<()> {
    let mut buf = vec![0u8; out.len() * 4];
    file.read_exact(&mut buf).context("failed to read f32 array")?;
    for (slot, chunk) in out.iter_mut().zip(buf.chunks_exact(4)) {
        *slot = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}
