#![feature(exit_status_error)]

use clap::Parser;
use dashmap::DashMap;
use libroxanne::common::Id;
use roxanne_analysis::export_index_sidecars;
use roxanne_analysis::Dataset;
use std::fs;
use std::fs::read;
use std::fs::read_to_string;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long)]
  m: usize,

  #[arg(long)]
  ef: usize,

  #[arg(long)]
  iter: usize,

  #[arg(long)]
  alpha: f64,

  #[arg(long)]
  batch: usize,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let variant = format!("vamanaaccel-{}M-{}ef-{}a", args.m, args.ef, args.alpha);
  let out = format!("dataset/{}/out/{}", ds.name, variant);
  fs::create_dir_all(&out).unwrap();

  Command::new("python")
    .arg("vamana-accel/main.py")
    .arg("--dtype")
    .arg("float32")
    .arg("--dim")
    .arg(ds.info.dim.to_string())
    .arg("--m")
    .arg(args.m.to_string())
    .arg("--ef")
    .arg(args.ef.to_string())
    .arg("--iter")
    .arg(args.iter.to_string())
    .arg("--alpha")
    .arg(args.alpha.to_string())
    .arg("--batch")
    .arg(args.batch.to_string())
    .arg("--out")
    .arg(format!("{out}/graph.msgpack"))
    .arg("--out-medoid")
    .arg(format!("{out}/medoid.txt"))
    .arg(format!("dataset/{}/vectors.bin", ds.name))
    .status()
    .unwrap()
    .exit_ok()
    .unwrap();

  let graph: DashMap<Id, Vec<Id>> =
    rmp_serde::from_slice(&read(format!("{out}/graph.msgpack")).unwrap()).unwrap();
  let medoid = read_to_string(format!("{out}/medoid.txt"))
    .unwrap()
    .parse::<Id>()
    .unwrap();
  export_index_sidecars(&ds, &variant, &graph, medoid);
}
