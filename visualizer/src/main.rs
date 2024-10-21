use ahash::AHashSet;
use itertools::Itertools;
use libroxanne::common::metric_euclidean;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaInstrumentationEvent;
use libroxanne::vamana::VamanaParams;
use ndarray::array;
use ordered_float::OrderedFloat;
use rand::thread_rng;
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::iter::zip;
use std::sync::Arc;

fn main() {
  let mut rng = thread_rng();
  let metric = metric_euclidean;
  // Let's plot points such that it fits comfortably spread across a widescreen display, useful for when we visualise this.
  let x_range = 0.0f32..1200.0f32;
  let y_range = 0.0f32..700.0f32;
  let n = 100u32;
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
  let dataset = zip(ids.clone(), points.clone()).collect_vec();

  let events: Arc<parking_lot::Mutex<Vec<VamanaInstrumentationEvent<f32>>>> = Default::default();

  let vamana = InMemoryVamana::init(
    dataset,
    metric,
    VamanaParams {
      beam_width: 1,
      degree_bound: r,
      distance_threshold: 1.1,
      insert_batch_size: 64,
      medoid_sample_size: 10_000,
      search_list_cap,
    },
    Some(Box::new({
      let events = events.clone();
      move |e| events.lock().push(e)
    })),
  );

  // Test k-NN of every point.
  let mut correct = 0;
  for a in ids.iter().cloned() {
    let a_pt = &points[a as usize];
    let truth = ids
      .iter()
      .cloned()
      .filter(|&b| b != a)
      .sorted_unstable_by_key(|&b| OrderedFloat(metric(&a_pt.view(), &points[b as usize].view())))
      .take(k)
      .collect::<AHashSet<_>>();
    let approx = vamana
      .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
      .into_iter()
      .map(|pd| pd.id)
      .filter(|&b| b != a)
      .take(k)
      .collect::<AHashSet<_>>();
    correct += approx.intersection(&truth).count();
  }
  println!(
    "[2D Pairwise] Correct: {}/{} ({:.2}%)",
    correct,
    k * n as usize,
    correct as f64 / (k * n as usize) as f64 * 100.0
  );

  #[derive(Serialize)]
  struct DataNode {
    id: u32,
    point: Vec<f32>,
    knn: Vec<u32>,
  }
  let nodes = ids
    .iter()
    .map(|&id| {
      let point = &points[id as usize];
      let knn = vamana
        .query(&point.view(), k)
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
    events: Vec<VamanaInstrumentationEvent<f32>>,
    medoid: u32,
    nodes: Vec<DataNode>,
  }

  serde_json::to_writer_pretty(File::create("visualizer.data.json").unwrap(), &Data {
    events: events.lock().to_vec(),
    medoid: vamana.medoid(),
    nodes,
  })
  .unwrap();
}
