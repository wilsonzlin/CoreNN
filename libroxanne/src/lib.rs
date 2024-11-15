#![feature(f16)]

pub mod common;
pub mod db;
pub mod hnsw;
pub mod in_memory;
pub mod pq;
pub mod queue;
pub mod search;
pub mod vamana;

pub struct RoxanneDbParams {
  // Query.
  pub query_search_list_cap: usize,
  // Corresponds to W in the DiskANN paper, section 3.3 (DiskANN Beam Search).
  pub beam_width: usize,
}
