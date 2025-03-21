use super::Compressor;
use bytemuck::cast_slice;
use half::f16;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;

pub struct TruncCompressor {
  dim: usize,
}

impl TruncCompressor {
  pub fn new(dim: usize) -> Self {
    Self { dim }
  }
}

impl Compressor for TruncCompressor {
  fn compress(&self, v: &ArrayView1<f16>) -> Vec<u8> {
    let s = v.slice(s![..self.dim]);
    let s = s.as_slice().unwrap();
    cast_slice(s).to_vec()
  }

  fn decompress(&self, v: &[u8]) -> Array1<f16> {
    // Callers should directly compare truncated vectors.
    // Re-expanding to original dims with padded zeros is unnecessary.
    let v: &[f16] = cast_slice(v);
    Array1::from(v.to_vec())
  }
}
