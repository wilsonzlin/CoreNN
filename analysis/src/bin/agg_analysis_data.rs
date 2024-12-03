use std::{fs, path::{Path, PathBuf}};

use std::collections::{HashMap};
use bytemuck::cast_slice;
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rayon::iter::{ParallelBridge, ParallelIterator};
use rmp_serde::to_vec_named;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

fn histogram(vals: &[f64], bins: usize) -> Vec<(f64, usize)> {
  let min = vals.iter().min_by_key(|&&v| OrderedFloat(v)).unwrap();
  let max = vals.iter().max_by_key(|&&v| OrderedFloat(v)).unwrap();
  let bin_size = (max - min) / bins as f64;
  let mut hist = vec![0usize; bins];
  for v in vals {
    let bin = (((v - min) / bin_size) as usize).min(bins - 1);
    hist[bin] += 1;
  }
  hist.into_iter().enumerate().map(|(i, c)| (min + i as f64 * bin_size, c)).collect_vec()
}

#[derive(Serialize)]
struct Variant {
  name: String,
  out_neighbor_count_hist: Vec<(f64, usize)>,
  edge_dist_hist: Vec<(f64, usize)>,
  medoid_dist_hist: Vec<(f64, usize)>,
  query_metrics_means: HashMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct QueryMetrics {
  iterations: Vec<HashMap<String, f64>>,
}

fn read_msgpack<T: DeserializeOwned>(p: impl AsRef<Path>) -> T {
  rmp_serde::from_slice(&fs::read(p).unwrap()).unwrap()
}

fn main() {
  let ds = std::env::var("DS").unwrap();
  let base_dir = PathBuf::from(format!("dataset/{ds}/out"));

  let variants = fs::read_dir(&base_dir)
    .unwrap()
    .par_bridge()
    .filter_map(|e| {
      let e = e.unwrap();
      let variant = e.file_name().into_string().unwrap();
      if variant.starts_with("_") {
        return None;
      };
      let d = base_dir.join(&variant);

      let graph: HashMap<usize, Vec<usize>> = read_msgpack(d.join("graph.msgpack"));
      let out_neighbor_counts = graph.values().map(|v| v.len() as f64).collect_vec();

      let medoid_dists: Vec<f32> = cast_slice(&fs::read(d.join("medoid_dists.mat")).unwrap()).to_vec();
      let medoid_dists = medoid_dists.iter().map(|&d| d as f64).collect_vec();

      let edge_dists_raw: HashMap<usize, HashMap<usize, f64>> = read_msgpack(d.join("edge_dists.msgpack"));
      let edge_dists = edge_dists_raw.values().flat_map(|d| d.values().cloned()).collect_vec();

      // query_metrics: an array of { iterations: Array<{ [metric_name: string]: number }> }, one for each query.
      let query_metrics: Vec<QueryMetrics> = read_msgpack(d.join("query_metrics.msgpack"));
      let mut query_metrics_means = HashMap::new();
      for metric_name in query_metrics[0].iterations[0].keys() {
        let arrays = query_metrics.iter().map(|q| q.iterations.iter().map(|it| it[metric_name]).collect_vec()).collect_vec();
        let max_len = arrays.iter().map(|a| a.len()).max().unwrap();
        let mut mean = vec![0.0f64; max_len];
        for i in 0..max_len {
          for a in arrays.iter() {
            mean[i] += *a.get(i).unwrap_or(a.last().unwrap());
          }
        }
        for e in mean.iter_mut() {
          *e /= arrays.len() as f64;
        }
        query_metrics_means.insert(metric_name.clone(), mean);
      }

      Some(Variant {
        name: variant,
        query_metrics_means,
        edge_dist_hist: histogram(&edge_dists, 150),
        medoid_dist_hist: histogram(&medoid_dists, 150),
        out_neighbor_count_hist: histogram(&out_neighbor_counts, 150),
      })
    })
    .collect::<Vec<_>>();

  fs::write(
    base_dir.join("_agg.msgpack"),
    to_vec_named(&variants).unwrap(),
  ).unwrap();
}
