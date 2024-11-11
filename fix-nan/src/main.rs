use clap::Parser;
use libroxanne::db::Db;
use libroxanne::db::NodeData;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,
}

// Inspired by https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html.
fn nan_to_num(v: f32) -> f32 {
  if v.is_nan() {
    0.0
  } else if v.is_infinite() {
    if v.is_sign_positive() {
      f32::MAX
    } else {
      -f32::MAX
    }
  } else {
    v
  }
}

fn main() {
  let args = Args::parse();

  let db = Db::open(&args.path);

  let mut fixed = 0;
  let mut total = 0;
  for (id, node) in db.iter_nodes() {
    if node.vector.iter().any(|v| !v.is_finite()) {
      fixed += 1;
      db.write_node(id, &NodeData {
        neighbors: node.neighbors,
        vector: node.vector.into_iter().map(|v| nan_to_num(v)).collect(),
      });
    }
    total += 1;
  }
  db.flush();
  println!("All done! Fixed {fixed} of {total} vectors.");
}
