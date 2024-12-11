#![feature(exit_status_error)]

use clap::Parser;
use libroxanne::common::Id;
use roxanne_analysis::export_index;
use roxanne_analysis::read_graph_matrix;
use roxanne_analysis::Dataset;
use std::fs;
use std::fs::read_to_string;

/// This program exports a vamana-accel built graph matrix and medoid, which should already be placed in the dataset/$DS/out/vamanaaccel-$M-$EF-$ALPHA directory.
/// To build a vamana-accel index instead, use analysis_build_vamana_accel.
/// This is useful if you ran the vamana-accel/main.py directly instead of using analysis_build_vamana_accel.
#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long)]
  m: usize,

  #[arg(long)]
  ef: usize,

  #[arg(long)]
  alpha: f64,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let variant = format!("vamanaaccel-{}M-{}ef-{}a", args.m, args.ef, args.alpha);
  let out = format!("dataset/{}/out/{}", ds.name, variant);
  fs::create_dir_all(&out).unwrap();

  let path_graph_mat = format!("{out}/graph.mat");
  let path_medoid = format!("{out}/medoid.txt");

  let graph = read_graph_matrix(path_graph_mat, (ds.info.n, args.m));
  println!("Loaded graph");

  let medoid = read_to_string(&path_medoid).unwrap().parse::<Id>().unwrap();

  export_index(&ds, &variant, &graph, medoid);
}
