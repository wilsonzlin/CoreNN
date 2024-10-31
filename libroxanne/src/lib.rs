use ndarray_linalg::Scalar;
use std::mem::size_of;

pub mod pq;
pub mod queue;
pub mod vamana;

pub struct RoxanneDbParams {
  pub n_per_shard: usize,
  pub overlap: usize, // How many shards to insert each point into.
  pub pq_subspaces: usize,
  pub shards: usize,
}

impl RoxanneDbParams {
  // memlimit: how much memory (in bytes) to allow the DB to use for indexing, and when the indexed DB is loaded in memory.
  // n: the maximum number of vectors this DB will ever have.
  pub fn recommended<DType: Scalar>(memlimit: usize, n: usize, dims: usize) -> Self {
    let n_per_shard = memlimit / (dims * size_of::<DType>());
    Self {
      n_per_shard,
      overlap: 2,
      pq_subspaces: memlimit / n,
      shards: n.div_ceil(n_per_shard),
    }
  }
}
