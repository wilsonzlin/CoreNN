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

/// Precomputed distance table for asymmetric distance computation (ADC).
/// Created once per query, reused for all distance computations.
pub type DistanceTable = Arc<dyn Any + Send + Sync>;

pub trait Compressor: Debug + Send + Sync {
  fn into_compressed(&self, v: VecData) -> CV;
  fn compress(&self, v: &VecData) -> CV {
    self.into_compressed(v.clone())
  }
  fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f64;
  
  /// Create a precomputed distance table for ADC (Asymmetric Distance Computation).
  /// This is called once per query and enables fast distance computation.
  /// Default implementation returns None (no ADC support).
  fn create_distance_table(&self, _query: &VecData, _metric: StdMetric) -> Option<DistanceTable> {
    None
  }
  
  /// Compute distance using a precomputed table (ADC).
  /// Returns None if ADC is not supported, in which case the caller should fall back to `dist`.
  fn dist_with_table(&self, _table: &DistanceTable, _cv: &CV) -> Option<f64> {
    None
  }
  
  /// Fast distance from a raw query to a compressed vector using ADC if available.
  /// Falls back to compressing the query and using symmetric distance.
  fn dist_query(&self, query: &VecData, cv: &CV, metric: StdMetric, table: Option<&DistanceTable>) -> f64 {
    if let Some(table) = table {
      if let Some(dist) = self.dist_with_table(table, cv) {
        return dist;
      }
    }
    // Fallback: compress query and compute symmetric distance
    self.dist(metric, &self.compress(query), cv)
  }
}

impl CacheableTransformer<CV> for Arc<dyn Compressor> {
  fn transform(&self, node: DbNodeData) -> CV {
    self.into_compressed(node.into_vector())
  }
}
