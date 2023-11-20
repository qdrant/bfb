use memmap2::{Mmap, MmapOptions};
use std::fs::OpenOptions;
use std::mem::transmute;
use std::path::Path;

pub struct FBinReader {
    pub num_vectors: i32,
    pub dim: i32,
    pub iter_offset: usize,
    pub header_size: usize,
    mmap: Mmap,
}

impl FBinReader {
    pub fn new(path: &Path) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .append(false)
            .create(false)
            .open(path)
            .unwrap();

        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let int_size = std::mem::size_of::<i32>();
        let dim_offset = int_size;
        let header_size = dim_offset + int_size;
        let num_vectors_raw = &mmap[0..dim_offset];
        let num_dim_raw = &mmap[dim_offset..header_size];

        let num_vectors = i32::from_le_bytes(num_vectors_raw.try_into().unwrap());
        let dim = i32::from_le_bytes(num_dim_raw.try_into().unwrap());

        FBinReader {
            num_vectors,
            dim,
            iter_offset: 0,
            header_size,
            mmap,
        }
    }

    pub fn read_vector(&self, idx: usize) -> &[f32] {
        let vector_size = self.dim as usize * std::mem::size_of::<f32>();
        let vector_offset = self.header_size + idx * vector_size;
        let vector_raw = &self.mmap[vector_offset..vector_offset + vector_size];
        let arr: &[f32] = unsafe { transmute(vector_raw) };
        &arr[..self.dim as usize]
    }
}

impl Iterator for FBinReader {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_offset >= self.num_vectors as usize {
            return None;
        }

        let vector = self.read_vector(self.iter_offset).to_vec();
        self.iter_offset += 1;
        Some(vector)
    }
}
