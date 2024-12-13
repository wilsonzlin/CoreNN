#![feature(exit_status_error)]

use clap::Parser;
use roxanne_analysis::randinit::export_randinit;
use roxanne_analysis::Dataset;

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
  iter: usize,

  #[arg(long)]
  alpha: f64,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let variant = format!(
    "randinit-{}M-{}Mmax-{}r-{}ef-{}iter-{}a",
    args.m, args.m_max, args.r, args.ef, args.iter, args.alpha
  );
  let out = format!("dataset/{}/out/{}", ds.name, variant);

  export_randinit(&ds, &out, &variant, args.m);
}
