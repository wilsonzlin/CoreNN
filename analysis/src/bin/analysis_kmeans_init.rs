use ahash::HashMap;
use ahash::HashSet;
use bytemuck::cast_slice;
use clap::Parser;
use dashmap::DashMap;
use half::f16;
use itertools::Itertools;
use libroxanne::common::Id;
use libroxanne::in_memory::calc_approx_medoid;
use libroxanne::in_memory::InMemoryIndex;
use libroxanne::vamana::VamanaParams;
use ndarray::Array2;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use roxanne_analysis::export_index;
use roxanne_analysis::new_pb;
use roxanne_analysis::Dataset;
use std::fs;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(clap::Parser)]
struct Args {
  /// Number of clusters for k-means
  #[arg(long)]
  k: usize,

  /// Maximum out-degree of each node
  #[arg(long)]
  degree_bound: usize,

  /// Decay factor for distance-based weighting
  #[arg(long)]
  decay: f64,
}

fn main() {
  let Args {
    k,
    degree_bound,
    decay,
  } = Args::parse();

  let ds = Dataset::init();
  let n = ds.info.n;

  let metric = ds.info.metric.get_fn::<f32>();

  let ids = (0..n).collect_vec();
  let mat = ds.read_vectors();
  let vecs = mat.rows().into_iter().map(|r| r.to_owned()).collect_vec();
  let id_to_vec = Arc::new(zip(ids.clone(), vecs.clone()).collect::<DashMap<_, _>>());

  // Use tools/calc_kmeans_gpu.py to build these data files.
  let kmeans_data_dir: PathBuf = format!("dataset/{}/kmeans/{}", ds.name, k).into();
  let dists_to_centroids = {
    let raw = fs::read(kmeans_data_dir.join("dists_to_centroids.f16.bin")).unwrap();
    let raw: &[f16] = cast_slice(&raw);
    Array2::from_shape_vec((n, k), raw.to_vec()).unwrap()
  };
  println!("Loaded vector distances to centroids");
  let cluster_members: HashMap<usize, Vec<Id>> =
    rmp_serde::from_slice(&fs::read(kmeans_data_dir.join("cluster_members.msgpack")).unwrap())
      .unwrap();
  assert_eq!(dists_to_centroids.shape()[1], cluster_members.len());
  println!("Effective clusters: {}", cluster_members.len());

  println!("Assigning edges");
  let graph = DashMap::<Id, HashSet<Id>>::new();
  let pb = new_pb(n);
  ids.par_iter().for_each(|&id| {
    let centroid_dists = dists_to_centroids.row(id);
    let weights = centroid_dists
      .iter()
      .map(|d| (-decay * d.to_f64()).exp())
      .collect_vec();
    let dist = WeightedIndex::new(&weights).unwrap();
    for _ in 0..degree_bound {
      let chosen_cluster = dist.sample(&mut thread_rng());
      let chosen_out_neighbor = *cluster_members[&chosen_cluster]
        .choose(&mut thread_rng())
        .unwrap();
      graph.entry(id).or_default().insert(chosen_out_neighbor);
      graph.entry(chosen_out_neighbor).or_default().insert(id);
    }
    pb.inc(1);
  });
  pb.finish();

  let params = VamanaParams {
    degree_bound,
    distance_threshold: 1.1,
    update_batch_size: 64,
    update_search_list_cap: degree_bound * 2,
  };

  println!("Calculating approx. medoid of {n} vectors");
  let medoid = calc_approx_medoid(&id_to_vec, metric, 10_000, None);

  let idx = InMemoryIndex {
    graph: graph.into_iter().map(|(k, v)| (k, v.into_iter().collect_vec())).collect::<DashMap<_, _>>().into(),
    medoid,
    metric,
    params,
    precomputed_dists: None,
    vectors: id_to_vec,
  };

  let out_dir = format!("kmeansinit-{}M-k{}-{}decay", degree_bound, k, decay);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();
  export_index(&ds, &out_dir, &idx.graph, idx.medoid);
}
