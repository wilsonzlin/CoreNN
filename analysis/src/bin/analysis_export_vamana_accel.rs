#![feature(exit_status_error)]

use bytemuck::cast_slice;
use clap::Parser;
use dashmap::DashMap;
use libroxanne::common::Id;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;
use std::fs::read;
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

  let raw = read(&path_graph_mat).unwrap();
  let flat: &[u32] = cast_slice(&raw);
  assert_eq!(flat.len(), ds.info.n * args.m);
  let graph: DashMap<Id, Vec<Id>> = DashMap::new();
  flat
    .into_par_iter()
    .chunks(args.m)
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
  println!("Loaded graph");

  let medoid = read_to_string(&path_medoid).unwrap().parse::<Id>().unwrap();

  export_index(&ds, &variant, &graph, medoid);
}
