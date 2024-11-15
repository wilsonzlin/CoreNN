use ahash::HashSet;
use itertools::Itertools;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use libroxanne::in_memory::InMemoryIndex;
use libroxanne::search::GreedySearchable;
use ndarray::array;
use ordered_float::OrderedFloat;
use rand::thread_rng;
use rand::Rng;
use serde::Serialize;
use std::fs::File;

fn main() {
  let mut rng = thread_rng();
  let metric = metric_euclidean::<f32>;
  // Let's plot points such that it fits comfortably spread across a widescreen display, useful for when we visualise this.
  let x_range = 0.0f32..1200.0f32;
  let y_range = 0.0f32..700.0f32;
  let n = 100usize;
  let r = 10;
  let ids = (0..n).collect_vec();
  let k = 10;
  let search_list_cap = k * 2;
  let points = (0..n)
    .map(|_| {
      array![
        rng.gen_range(x_range.clone()),
        rng.gen_range(y_range.clone()),
      ]
    })
    .collect_vec();

  let vamana = InMemoryIndex::builder(ids.clone(), points.clone())
    .metric(metric)
    .degree_bound(r)
    .update_search_list_cap(search_list_cap)
    .build();

  // Test k-NN of every point.
  let mut correct = 0;
  for a in ids.iter().cloned() {
    let a_pt = &points[a];
    let truth = ids
      .iter()
      .cloned()
      .filter(|&b| b != a)
      .sorted_unstable_by_key(|&b| OrderedFloat(metric(&a_pt.view(), &points[b].view())))
      .take(k)
      .collect::<HashSet<_>>();
    let approx = vamana
      .query_builder(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
      .query()
      .into_iter()
      .map(|pd| pd.id)
      .filter(|&b| b != a)
      .take(k)
      .collect::<HashSet<_>>();
    correct += approx.intersection(&truth).count();
  }
  println!(
    "[2D Pairwise] Correct: {}/{} ({:.2}%)",
    correct,
    k * n,
    correct as f64 / (k * n) as f64 * 100.0
  );

  #[derive(Serialize)]
  struct DataNode {
    id: Id,
    point: Vec<f32>,
    knn: Vec<Id>,
  }
  let nodes = ids
    .iter()
    .map(|&id| {
      let point = &points[id];
      let knn = vamana
        .query_builder(&point.view(), k)
        .query()
        .into_iter()
        .map(|pd| pd.id)
        .collect();
      DataNode {
        id,
        point: point.to_vec(),
        knn,
      }
    })
    .collect::<Vec<_>>();

  #[derive(Serialize)]
  struct Data {
    medoid: Id,
    nodes: Vec<DataNode>,
  }

  serde_json::to_writer_pretty(File::create("visualizer.data.json").unwrap(), &Data {
    medoid: vamana.medoid,
    nodes,
  })
  .unwrap();
}
