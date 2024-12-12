pub mod randinit;

use ahash::HashMap;
use ahash::HashSet;
use bytemuck::cast_slice;
use bytemuck::Pod;
use dashmap::DashMap;
use half::f16;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use libroxanne::common::StdMetric;
use libroxanne::search::GreedySearchParams;
use libroxanne::search::GreedySearchableSync;
use libroxanne::search::Query;
use libroxanne::search::SearchMetrics;
use ndarray::Array2;
use ndarray::ArrayView2;
use num::ToPrimitive;
use ordered_float::OrderedFloat;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
pub enum DatasetDtype {
  F16,
  F32,
  F64,
  U8,
  U16,
  U32,
  U64,
  I8,
  I16,
  I32,
  I64,
}

#[derive(Deserialize)]
pub struct DatasetInfo {
  pub dtype: DatasetDtype,
  pub metric: StdMetric,
  pub dim: usize,
  pub n: usize,
  pub q: usize,
  pub k: usize,
}

pub struct Dataset {
  pub name: String,
  pub dir: String,
  pub info: DatasetInfo,
}

fn read_raw<T: Pod>(path: impl AsRef<Path>) -> Vec<T> {
  let t_size = size_of::<T>();
  let mut f = File::open(path).unwrap();
  let f_size = f.metadata().unwrap().len() as usize;
  assert_eq!(f_size % t_size, 0);
  let mut arr = Vec::new();
  loop {
    let mut raw = vec![0u8; 1024 * 1024 * 1024];
    let n = f.read(&mut raw).unwrap();
    if n == 0 {
      break;
    };
    raw.truncate(n);
    // TODO Support interrupted/partial reads.
    assert_eq!(raw.len() % t_size, 0);
    let as_t: &[T] = cast_slice(&raw);
    arr.extend_from_slice(as_t);
  }
  arr
}

impl Dataset {
  pub fn init() -> Self {
    let name = std::env::var("DS").unwrap();
    let dir = format!("dataset/{name}");
    let info_raw = fs::read_to_string(format!("{dir}/info.toml")).unwrap();
    let info: DatasetInfo = toml::from_str(&info_raw).unwrap();
    // TODO Anything other than f32 for vectors is currently unsupported, due to Rust generics complexity.
    assert_eq!(info.dtype, DatasetDtype::F32);
    Dataset { name, dir, info }
  }

  pub fn read_vectors(&self) -> Array2<f32> {
    let path = format!("{}/vectors.bin", self.dir);
    let raw = read_raw::<f32>(path);
    Array2::from_shape_vec((self.info.n, self.info.dim), raw).unwrap()
  }

  pub fn read_dists(&self) -> Vec<f16> {
    let path = format!("{}/dists.bin", self.dir);
    read_raw::<f16>(path)
  }

  pub fn read_queries(&self) -> Array2<f32> {
    let path = format!("{}/queries.bin", self.dir);
    let raw = read_raw::<f32>(path);
    Array2::from_shape_vec((self.info.q, self.info.dim), raw).unwrap()
  }

  pub fn read_results(&self) -> Array2<u32> {
    let path = format!("{}/results.bin", self.dir);
    let raw = read_raw::<u32>(path);
    Array2::from_shape_vec((self.info.q, self.info.k), raw).unwrap()
  }
}

pub struct Eval {
  pub correct: usize,
  pub total: usize,
  pub query_metrics: Vec<SearchMetrics>,
}

impl Eval {
  pub fn ratio(&self) -> f64 {
    self.correct as f64 / self.total as f64
  }
}

pub fn eval<'a, 'g: 'a, G: GreedySearchableSync<f32, f32> + Send + Sync>(
  index: &'g G,
  queries: &ArrayView2<f32>,
  ground_truth: &ArrayView2<u32>,
  search_list_cap: usize,
  beam_width: usize,
) -> Eval {
  let q = queries.shape()[0];
  let k = ground_truth.shape()[1];
  let correct = AtomicUsize::new(0);
  let query_metrics = (0..q)
    .into_par_iter()
    .map(|i| {
      let knn_expected = HashSet::from_iter(ground_truth.row(i).mapv(|v| v as Id));
      let mut metrics = SearchMetrics::default();
      let res = index.greedy_search_sync(GreedySearchParams {
        query: Query::Vec(&queries.row(i)),
        k,
        search_list_cap,
        beam_width,
        start: index.medoid(),
        filter: |_| true,
        out_visited: None,
        out_metrics: Some(&mut metrics),
        ground_truth: Some(&knn_expected),
      });
      let knn_got = res.into_iter().map(|pd| pd.id).collect::<HashSet<_>>();
      correct.fetch_add(
        knn_expected.intersection(&knn_got).count(),
        Ordering::Relaxed,
      );
      metrics
    })
    .collect();

  Eval {
    correct: correct.load(Ordering::Relaxed),
    query_metrics,
    total: q * k,
  }
}

pub fn read_graph_matrix(path: impl AsRef<Path>, (n, m): (usize, usize)) -> DashMap<Id, Vec<Id>> {
  let raw = fs::read(path).unwrap();
  let flat: &[u32] = cast_slice(&raw);
  assert_eq!(flat.len(), n * m);
  let graph: DashMap<Id, Vec<Id>> = DashMap::new();
  flat
    .into_par_iter()
    .chunks(m)
    .enumerate()
    .for_each(|(id, row)| {
      // NULL_ID is i32::MAX.
      graph.insert(
        id,
        row
          .into_iter()
          .cloned()
          .filter(|&v| v != i32::MAX as u32)
          .map(|v| v as Id)
          .collect(),
      );
    });
  graph
}

pub fn export_index_sidecars(
  ds: &Dataset,
  out_dir: &str,
  graph: &DashMap<Id, Vec<Id>>,
  medoid: Id,
) {
  let vecs = ds.read_vectors();

  let ann_dists = graph
    .par_iter()
    .map(|e| {
      let i = *e.key();
      let dists = e
        .value()
        .par_iter()
        .map(|&j| {
          let dist = metric_euclidean(&vecs.row(i), &vecs.row(j));
          (j, dist)
        })
        .collect::<HashMap<_, _>>();
      (i, dists)
    })
    .collect::<HashMap<_, _>>();
  fs::write(
    format!("dataset/{}/out/{out_dir}/edge_dists.msgpack", ds.name),
    rmp_serde::to_vec_named(&ann_dists).unwrap(),
  )
  .unwrap();

  let medoid_vec = vecs.row(medoid);
  let medoid_dists = (0..vecs.shape()[0])
    .into_par_iter()
    .map(|i| metric_euclidean(&medoid_vec, &vecs.row(i)))
    .collect::<Vec<_>>();
  File::create(format!(
    "dataset/{}/out/{out_dir}/medoid_dists.mat",
    ds.name
  ))
  .unwrap()
  .write_all(
    &medoid_dists
      .into_iter()
      .flat_map(|d| (d as f32).to_le_bytes())
      .collect_vec(),
  )
  .unwrap();
}

pub fn export_index(ds: &Dataset, out_dir: &str, graph: &DashMap<Id, Vec<Id>>, medoid: Id) {
  fs::write(
    format!("dataset/{}/out/{out_dir}/graph.msgpack", ds.name),
    rmp_serde::to_vec_named(&graph).unwrap(),
  )
  .unwrap();
  fs::write(
    format!("dataset/{}/out/{out_dir}/medoid.txt", ds.name),
    medoid.to_string(),
  )
  .unwrap();

  export_index_sidecars(ds, out_dir, graph, medoid);
}

pub fn new_pb_with_template(len: impl ToPrimitive, template: &'static str) -> ProgressBar {
  let pb = ProgressBar::new(len.to_u64().unwrap());
  pb.set_style(
    ProgressStyle::with_template(template)
      .unwrap()
      .progress_chars("#>-"),
  );
  pb
}

/// Create a new progress bar that will show a custom message instead of the progress ratio.
/// This custom message can be set using `.set_message(...)`.
pub fn new_pb_with_msg(len: impl ToPrimitive) -> ProgressBar {
  new_pb_with_template(
    len,
    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} ({eta})",
  )
}

/// Create a new progress bar.
pub fn new_pb(len: impl ToPrimitive) -> ProgressBar {
  new_pb_with_template(
    len,
    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
  )
}

pub fn histogram(vals: &[f64], bins: usize) -> Vec<(f64, usize)> {
  let min = vals.iter().min_by_key(|&&v| OrderedFloat(v)).unwrap();
  let max = vals.iter().max_by_key(|&&v| OrderedFloat(v)).unwrap();
  let bin_size = (max - min) / bins as f64;
  let mut hist = vec![0usize; bins];
  for v in vals {
    let bin = (((v - min) / bin_size) as usize).min(bins - 1);
    hist[bin] += 1;
  }
  hist
    .into_iter()
    .enumerate()
    .map(|(i, c)| (min + i as f64 * bin_size, c))
    .collect_vec()
}
