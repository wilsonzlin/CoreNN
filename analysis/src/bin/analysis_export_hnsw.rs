use ahash::HashMap;
use ahash::HashMapExt;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use libroxanne_search::Id;
use ndarray::Array1;
use roxanne_analysis::analyse_index;
use roxanne_analysis::read_vectors;
use std::fs;
use std::fs::File;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long, default_value_t = 1.33)]
  search_list_cap_mul: f64,
}

fn main() {
  let args = Args::parse();

  fs::create_dir_all("out/hnsw").unwrap();

  let knns = read_vectors("groundtruth.ivecs", LittleEndian::read_u32_into);
  let k = knns[0].1.len();

  let hnsw = HnswIndex::load(128, File::open("out/hnsw/index.hnsw").unwrap());
  println!("Loaded index");

  let mut graph_dists_by_level = HashMap::<usize, HashMap<Id, HashMap<Id, f64>>>::new();
  for id in hnsw.labels() {
    for level in 0..=hnsw.get_node_level(id) {
      graph_dists_by_level.entry(level).or_default().insert(
        id,
        hnsw
          .get_level_neighbors(id, level)
          .map(|neighbor| {
            (
              neighbor,
              metric_euclidean(
                &Array1::from_vec(hnsw.get_data_by_label(id)).view(),
                &Array1::from_vec(hnsw.get_data_by_label(neighbor)).view(),
              ),
            )
          })
          .collect(),
      );
    }
  }
  println!("Calculated edge dists by level");
  fs::write(
    format!("out/hnsw/edge_dists_by_level.msgpack"),
    rmp_serde::to_vec_named(&graph_dists_by_level).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists by level");

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: hnsw.m,
    distance_threshold: 1.1,
    query_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
    update_batch_size: num_cpus::get(),
    update_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };
  let ds = InMemoryVamana::new(
    hnsw
      .labels()
      .map(|id| (id, hnsw.get_merged_neighbors(id, 0).into_iter().collect()))
      .collect::<DashMap<_, _>>(),
    hnsw
      .labels()
      .map(|id| (id, Array1::from_vec(hnsw.get_data_by_label(id))))
      .collect::<DashMap<_, _>>(),
  );
  let index = Vamana::new(ds, metric_euclidean, hnsw.entry_label(), params);
  println!("Built graph");

  analyse_index("hnsw", &index);
}
