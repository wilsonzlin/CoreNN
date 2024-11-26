use crate::common::StdMetric;
use serde::Deserialize;
use serde::Serialize;

fn default_beam_width() -> usize {
  4
}

fn default_degree_bound() -> usize {
  80
}

fn default_distance_threshold() -> f64 {
  1.1
}

fn default_max_degree_bound() -> usize {
  160
}

fn default_merge_threshold_additional_edges() -> usize {
  // Approx. 8 GB of RAM.
  1_000_000_000
}

fn default_merge_threshold_deletes() -> usize {
  // Approx. 3% of 1 billion.
  30_000_000
}

fn default_pq_sample_size() -> usize {
  100_000
}

fn default_query_search_list_cap() -> usize {
  160
}

fn update_batch_size() -> usize {
  // Enough to saturate 100-core machine.
  1024
}

fn update_search_list_cap() -> usize {
  // Double the query's.
  320
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RoxanneDbCfg {
  #[serde(default = "default_beam_width")]
  pub beam_width: usize,
  #[serde(default = "default_degree_bound")]
  pub degree_bound: usize,
  pub dim: usize,
  #[serde(default = "default_distance_threshold")]
  pub distance_threshold: f64,
  #[serde(default = "default_max_degree_bound")]
  pub max_degree_bound: usize,
  #[serde(default = "default_merge_threshold_additional_edges")]
  pub merge_threshold_additional_edges: usize,
  #[serde(default = "default_merge_threshold_deletes")]
  pub merge_threshold_deletes: usize,
  pub metric: StdMetric,
  #[serde(default = "default_pq_sample_size")]
  pub pq_sample_size: usize,
  pub pq_subspaces: usize,
  #[serde(default = "default_query_search_list_cap")]
  pub query_search_list_cap: usize,
  #[serde(default = "update_batch_size")]
  pub update_batch_size: usize,
  #[serde(default = "update_search_list_cap")]
  pub update_search_list_cap: usize,
}

impl Default for RoxanneDbCfg {
  /// This should only be used to complete optional fields (e.g. `..Default.default()`); required fields will be initialized to garbage values and must not be used.
  fn default() -> Self {
    Self {
      beam_width: default_beam_width(),
      degree_bound: default_degree_bound(),
      dim: 0,
      distance_threshold: default_distance_threshold(),
      max_degree_bound: default_max_degree_bound(),
      merge_threshold_additional_edges: default_merge_threshold_additional_edges(),
      merge_threshold_deletes: default_merge_threshold_deletes(),
      metric: StdMetric::L2,
      pq_sample_size: default_pq_sample_size(),
      pq_subspaces: 0,
      query_search_list_cap: default_query_search_list_cap(),
      update_batch_size: update_batch_size(),
      update_search_list_cap: update_search_list_cap(),
    }
  }
}
