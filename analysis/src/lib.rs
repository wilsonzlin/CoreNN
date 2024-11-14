#![feature(f16)]

use ahash::HashMap;
use ahash::HashSet;
use bytemuck::cast_slice;
use bytemuck::Pod;
use dashmap::DashMap;
use itertools::Itertools;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne_search::metric_euclidean;
use libroxanne_search::Id;
use libroxanne_search::SearchMetrics;
use ndarray::Array2;
use ndarray::ArrayView2;
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

pub fn eval<DS: VamanaDatastore<f32>>(
  index: &Vamana<f32, DS>,
  queries: &ArrayView2<f32>,
  ground_truth: &ArrayView2<u32>,
) -> Eval {
  let q = queries.shape()[0];
  let k = ground_truth.shape()[1];
  let correct = AtomicUsize::new(0);
  let query_metrics = (0..q)
    .into_par_iter()
    .map(|i| {
      let knn_expected = HashSet::from_iter(ground_truth.row(i).mapv(|v| v as Id));
      let (res, metrics) = index.query_with_metrics(&queries.row(i), k, Some(&knn_expected));
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

pub fn export_index(ds: &Dataset, out_dir: &str, graph: &DashMap<Id, Vec<Id>>, medoid: Id) {
  let vecs = ds.read_vectors();

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
  println!("Exported graph");

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
  println!("Calculated edge dists");
  fs::write(
    format!("dataset/{}/out/{out_dir}/edge_dists.msgpack", ds.name),
    rmp_serde::to_vec_named(&ann_dists).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists");

  let medoid_vec = vecs.row(medoid);
  let medoid_dists = (0..vecs.shape()[0])
    .into_par_iter()
    .map(|i| metric_euclidean(&medoid_vec, &vecs.row(i)))
    .collect::<Vec<_>>();
  println!("Calculated medoid dists");
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
  println!("Exported medoid dists");
}
