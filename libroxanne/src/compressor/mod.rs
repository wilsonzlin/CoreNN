use half::f16;
use ndarray::Array1;
use ndarray::ArrayView1;

pub mod pq;
pub mod trunc;

pub trait Compressor: Send + Sync {
  // These are &[u8] as different compressors use different representations, and it's simpler than generics.
  // Codebase philosophy applies here:
  // There are ways to add complexity here to avoid some malloc/memcpy across diverse implementations and usages, but I'd rather not.
  fn compress(&self, v: &ArrayView1<f16>) -> Vec<u8>;
  fn decompress(&self, v: &[u8]) -> Array1<f16>;
}
