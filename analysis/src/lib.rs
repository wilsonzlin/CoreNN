use ahash::HashMap;
use ahash::HashSet;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use itertools::Itertools;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne_search::metric_euclidean;
use libroxanne_search::Id;
use libroxanne_search::SearchMetrics;
use ndarray::Array1;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::mem::size_of;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

pub fn read_vectors<T: Copy + Default + Send, Reader: Fn(&[u8], &mut [T]) + Sync>(
  filename: &str,
  reader: Reader,
) -> Vec<(Id, Array1<T>)> {
  let raw = std::fs::read(format!("dataset/{filename}")).unwrap();
  let dims = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;

  let bytes_per_vec = 4 + dims * size_of::<T>();
  assert_eq!(raw.len() % bytes_per_vec, 0);
  let n = raw.len() / bytes_per_vec;

  (0..n)
    .into_par_iter()
    .map(|i| {
      let start = i * bytes_per_vec;
      let end = (i + 1) * bytes_per_vec;
      let raw = &raw[start..end];
      let (dims_raw, raw) = raw.split_at(4);
      assert_eq!(
        u32::from_le_bytes(dims_raw.try_into().unwrap()) as usize,
        dims
      );
      let mut vec = vec![T::default(); dims];
      reader(raw, vec.as_mut_slice());
      (i, Array1::from_vec(vec))
    })
    .collect::<Vec<_>>()
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
  queries: &[(usize, Array1<f32>)],
  ground_truth: &[(usize, Array1<u32>)],
) -> Eval {
  let k = ground_truth[0].1.len();
  let correct = AtomicUsize::new(0);
  let query_metrics = queries
    .into_par_iter()
    .zip(ground_truth)
    .map(|((_, vec), (_, knn_expected))| {
      let knn_expected = HashSet::from_iter(knn_expected.mapv(|v| v as Id));
      let (res, metrics) = index.query_with_metrics(&vec.view(), k, Some(&knn_expected));
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
    total: queries.len() * k,
  }
}

pub fn analyse_index(out_dir: &str, index: &Vamana<f32, InMemoryVamana<f32>>) {
  let vecs = read_vectors("base.fvecs", LittleEndian::read_f32_into);
  let qs = read_vectors("query.fvecs", LittleEndian::read_f32_into);
  let knns = read_vectors("groundtruth.ivecs", LittleEndian::read_u32_into);

  let graph = index.datastore().graph();
  fs::write(
    format!("out/{out_dir}/graph.msgpack"),
    rmp_serde::to_vec_named(&graph).unwrap(),
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
          let dist = metric_euclidean(&vecs[i].1.view(), &vecs[j].1.view());
          (j, dist)
        })
        .collect::<HashMap<_, _>>();
      (i, dists)
    })
    .collect::<HashMap<_, _>>();
  println!("Calculated edge dists");
  fs::write(
    format!("out/{out_dir}/edge_dists.msgpack"),
    rmp_serde::to_vec_named(&ann_dists).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists");

  let medoid_dists = vecs
    .par_iter()
    .map(|(_, vec_i)| metric_euclidean(&vecs[index.medoid()].1.view(), &vec_i.view()))
    .collect::<Vec<_>>();
  println!("Calculated medoid dists");
  File::create(format!("out/{out_dir}/medoid_dists.mat"))
    .unwrap()
    .write_all(
      &medoid_dists
        .into_iter()
        .flat_map(|d| (d as f32).to_le_bytes())
        .collect_vec(),
    )
    .unwrap();
  println!("Exported medoid dists");

  let e = eval(&index, &qs, &knns);
  fs::write(
    format!("out/{out_dir}/query_metrics.msgpack"),
    rmp_serde::to_vec_named(&e.query_metrics).unwrap(),
  )
  .unwrap();
  println!("Exported query metrics");

  println!(
    "Correct: {:.2}% ({}/{})",
    e.ratio() * 100.0,
    e.correct,
    e.total,
  );
}
