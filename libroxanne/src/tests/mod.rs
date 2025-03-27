use crate::common::Id;
use crate::db::NodeData;
use crate::Roxanne;
use half::f16;
use itertools::Itertools;
use ndarray::Array1;
use std::fs::create_dir_all;
use std::fs::remove_dir_all;
use std::sync::Arc;

mod cache;
mod nodes;

async fn create_test_rox(dir: &'static str) -> Arc<Roxanne> {
  create_dir_all(".testout").unwrap();
  let dir = format!(".testout/{dir}");
  let _ = remove_dir_all(&dir);
  Roxanne::open(&dir).await
}

fn node(neighbors: Vec<Id>, vector: Vec<f32>) -> NodeData {
  NodeData {
    neighbors,
    vector: Array1::from(vector.into_iter().map(f16::from_f32).collect_vec()),
  }
}
