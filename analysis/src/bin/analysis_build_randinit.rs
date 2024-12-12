#![feature(exit_status_error)]

use clap::Parser;
use roxanne_analysis::randinit::export_randinit;
use roxanne_analysis::Dataset;
use std::fs::create_dir_all;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long)]
  m: usize,

  #[arg(long)]
  m_max: usize,

  #[arg(long)]
  ef: usize,

  #[arg(long)]
  iter: usize,

  #[arg(long)]
  alpha: f64,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let variant = format!(
    "randinit-{}M-{}Mmax-{}ef-{}iter-{}a",
    args.m, args.m_max, args.ef, args.iter, args.alpha
  );
  let out = format!("dataset/{}/out/{}", ds.name, variant);
  create_dir_all(&out).unwrap();

  Command::new("python")
    .arg("roxanne-accel/randinit.py")
    .arg("--dtype")
    .arg("float32")
    .arg("--dim")
    .arg(ds.info.dim.to_string())
    .arg("--m")
    .arg(args.m.to_string())
    .arg("--m-max")
    .arg(args.m_max.to_string())
    .arg("--ef")
    .arg(args.ef.to_string())
    .arg("--iter")
    .arg(args.iter.to_string())
    .arg("--alpha")
    .arg(args.alpha.to_string())
    .arg("--out")
    .arg(format!("{out}/graph.mat"))
    .arg("--out-levels")
    .arg(format!("{out}/level_graphs.msgpack"))
    .arg("--out-medoid")
    .arg(format!("{out}/medoid.txt"))
    .arg(format!("dataset/{}/vectors.bin", ds.name))
    .status()
    .unwrap()
    .exit_ok()
    .unwrap();

  export_randinit(&ds, &out, &variant, args.m);
}
