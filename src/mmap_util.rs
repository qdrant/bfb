//! Helpers for memory-mapped dataset files.

use memmap2::{Advice, Mmap};

/// Fault every page of `mmap` into the process page cache.
///
/// Dataset readers serve vectors from an mmap. Opening the mapping alone does
/// not touch the bytes, so the first timed upload batch pays for cold disk I/O
/// (painful on FUSE/NTFS data disks). Call this during reader construction —
/// before any upload timer starts — so `upload_time` measures ingest, not
/// dataset reads.
pub fn prefault_mmap(mmap: &Mmap) {
    // Best-effort hints; ignore errors (not all kernels/FS honor them).
    let _ = mmap.advise(Advice::Sequential);
    let _ = mmap.advise(Advice::WillNeed);

    // Force the faults now. Touch one byte per page so a cold 614 MB
    // `vectors.npy` is fully resident before the timed upsert loop.
    const PAGE: usize = 4096;
    let bytes: &[u8] = mmap;
    let mut checksum = 0u8;
    let mut offset = 0;
    while offset < bytes.len() {
        checksum ^= bytes[offset];
        offset += PAGE;
    }
    if let Some(&last) = bytes.last() {
        checksum ^= last;
    }
    std::hint::black_box(checksum);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn prefault_reads_every_page_without_changing_bytes() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let payload: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        file.write_all(&payload).unwrap();
        file.flush().unwrap();

        let mmap = unsafe { Mmap::map(file.as_file()) }.unwrap();
        assert_eq!(&mmap[..], payload.as_slice());
        prefault_mmap(&mmap);
        assert_eq!(&mmap[..], payload.as_slice());
    }
}
