use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use libroxanne::db::Db;
use libroxanne::db::NodeData;
use libroxanne::pq::ProductQuantizer;
use libroxanne_search::Id;
use ndarray::Array2;
use rand::thread_rng;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use std::path::PathBuf;
use std::str::from_utf8;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to Roxanne index.
  #[arg()]
  path: PathBuf,

  /// Subspaces.
  #[arg(long)]
  subspaces: usize,

  /// Sample probability between 0 and 1 inclusive.
  #[arg(long)]
  sample: f64,
}

fn new_pb(len: usize) -> ProgressBar {
  let pb = ProgressBar::new(len.try_into().unwrap());
  pb.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-"),
  );
  pb
}

// ProductQuantizer uses K-means, which breaks with NaN values. Therefore, we replace all NaN elements in all embeddings with zero.
// Inspired by https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html.
fn nan_to_num(src: impl IntoIterator<Item = f32>) -> impl Iterator<Item = f32> {
  src.into_iter().map(|v| {
    if v.is_nan() {
      0.0
    } else if v.is_infinite() {
      if v.is_sign_positive() {
        f32::MAX
      } else {
        f32::MIN
      }
    } else {
      v
    }
  })
}

fn main() {
  let args = Args::parse();

  let db = Db::open(&args.path);
  let dim = db.read_dim();
  let dim_pq = args.subspaces;

  let mut n = 0;
  let mut n_sample = 0;
  let mut mat_raw = Vec::<f32>::new();
  for e in db
    .inner()
    .full_iterator(IteratorMode::From(b"node/", Direction::Forward))
  {
    let (k, v) = e.unwrap();
    if !k.starts_with(b"node/") {
      break;
    };
    n += 1;
    if !thread_rng().gen_bool(args.sample) {
      continue;
    };
    let node = NodeData::deserialize(&v);
    assert_eq!(node.vector.len(), dim);
    mat_raw.extend(nan_to_num(node.vector));
    n_sample += 1;
  }
  let mat = Array2::from_shape_vec((n_sample, dim), mat_raw).unwrap();
  println!("Sampled {n_sample} vectors");
  println!(
    "PQ model: reduce {dim}-dim ({:.2} GB) to {dim_pq}-dim ({:.2} GB)",
    (n * dim * 4) as f64 / 1024.0 / 1024.0 / 1024.0,
    (n * dim_pq) as f64 / 1024.0 / 1024.0 / 1024.0
  );

  let pq = ProductQuantizer::train(&mat.view(), args.subspaces);
  db.write_pq_model(&pq);
  println!("Trained PQ");

  let pb = new_pb(n);
  for batch in &db
    .inner()
    .full_iterator(IteratorMode::From(b"node/", Direction::Forward))
    .map(|e| e.unwrap())
    .take_while(|e| e.0.starts_with(b"node/"))
    .chunks(256)
  {
    let mut ids = Vec::new();
    let mut mat_raw = Vec::new();
    for (k, v) in batch {
      let id_raw = k.strip_prefix(b"node/").unwrap();
      let id = Id::from_str_radix(from_utf8(id_raw).unwrap(), 10).unwrap();
      ids.push(id);
      let node = NodeData::deserialize(&v);
      mat_raw.extend(nan_to_num(node.vector));
    }
    let mat = Array2::from_shape_vec((ids.len(), dim), mat_raw).unwrap();
    let mat_pq = pq.encode(&mat.view());
    ids.into_par_iter().enumerate().for_each(|(i, id)| {
      let vec_pq = mat_pq.row(i).to_vec();
      db.write_pq_vec(id, vec_pq);
      pb.inc(1);
    });
  }
  pb.finish();
  println!("Finalizing database");

  db.flush();
  println!("All done!");
}
