#![feature(exit_status_error)]

use clap::Parser;
use roxanne_analysis::analyze::analyze_graph;
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
  r: usize,

  #[arg(long)]
  ef: usize,

  #[arg(long)]
  it: String,

  #[arg(long)]
  alpha: f64,

  #[arg(long)]
  batch: Option<usize>,

  #[arg(long)]
  nn_samp: Option<usize>,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let variant = format!(
    "randinit-{}M-{}Mmax-{}r-{}ef-{}it-{}a",
    args.m, args.m_max, args.r, args.ef, args.it, args.alpha
  );
  let out = format!("dataset/{}/out/{}", ds.name, variant);
  create_dir_all(&out).unwrap();

  let mut cmd = Command::new("python");
  cmd
    .arg("roxanne-accel/randinit.py")
    .arg("--dtype")
    .arg("float32")
    .arg("--dim")
    .arg(ds.info.dim.to_string())
    .arg("--m")
    .arg(args.m.to_string())
    .arg("--m-max")
    .arg(args.m_max.to_string())
    .arg("--r")
    .arg(args.r.to_string())
    .arg("--ef")
    .arg(args.ef.to_string())
    .arg("--it")
    .arg(args.it)
    .arg("--alpha")
    .arg(args.alpha.to_string())
    .arg("--out")
    .arg(format!("{out}/graph.mat"))
    .arg("--out-levels")
    .arg(format!("{out}/level_graphs.msgpack"))
    .arg("--out-medoid")
    .arg(format!("{out}/medoid.txt"))
    .arg("--eval-q")
    .arg(format!("dataset/{}/queries.bin", ds.name))
    .arg("--eval-r")
    .arg(format!("dataset/{}/results.bin", ds.name))
    .arg(format!("dataset/{}/vectors.bin", ds.name));
  if let Some(batch) = args.batch {
    cmd.arg("--batch").arg(batch.to_string());
  }
  if let Some(nn_samp) = args.nn_samp {
    cmd.arg("--nn-samp").arg(nn_samp.to_string());
  }
  cmd.status().unwrap().exit_ok().unwrap();

  export_randinit(&ds, &out, &variant, args.m);

  analyze_graph(&ds, &variant, 1, ds.info.k, None);
}
