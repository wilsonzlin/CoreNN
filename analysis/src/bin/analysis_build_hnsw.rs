#![feature(exit_status_error)]

use ahash::HashMap;
use ahash::HashMapExt;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use ndarray::Array1;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;
use std::fs::File;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long)]
  m: usize,

  #[arg(long)]
  ef: usize,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let out_dir = format!("hnsw-{}M-{}ef", args.m, args.ef);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  Command::new("python")
    .arg("analysis/src/build_hnsw.py")
    .env("OUT", out_dir.clone())
    .env("DIM", ds.info.dim.to_string())
    .env("M", args.m.to_string())
    .env("EF", args.ef.to_string())
    .status()
    .unwrap()
    .exit_ok()
    .unwrap();

  let hnsw = HnswIndex::load(
    ds.info.dim,
    File::open(format!("dataset/{}/out/{out_dir}/index.hnsw", ds.name)).unwrap(),
  );
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
    format!(
      "dataset/{}/out/{out_dir}/edge_dists_by_level.msgpack",
      ds.name
    ),
    rmp_serde::to_vec_named(&graph_dists_by_level).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists by level");

  export_index(
    &ds,
    &out_dir,
    &hnsw
      .labels()
      .map(|id| (id, hnsw.get_merged_neighbors(id, 0).into_iter().collect()))
      .collect::<DashMap<_, _>>(),
    hnsw.entry_label(),
  );
}
