use crate::cache::CacheableTransformer;
use crate::metric::StdMetric;
use crate::store::schema::DbNodeData;
use crate::vec::VecData;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

pub mod pq;
pub mod trunc;

// Compressed vector.
pub type CV = Arc<dyn Any + Send + Sync>;

pub trait Compressor: Debug + Send + Sync {
  fn into_compressed(&self, v: VecData) -> CV;
  fn compress(&self, v: &VecData) -> CV {
    self.into_compressed(v.clone())
  }
  fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f64;
}

impl CacheableTransformer<CV> for Arc<dyn Compressor> {
  fn transform(&self, node: DbNodeData) -> CV {
    self.into_compressed(node.into_vector())
  }
}
