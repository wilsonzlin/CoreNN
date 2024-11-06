use byteorder::ByteOrder;
use byteorder::LittleEndian;
use itertools::Itertools;
use libroxanne_search::metric_euclidean;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use roxanne_analysis::read_vectors;
use std::fs;
use std::fs::File;
use std::io::Write;

fn main() {
  let ds = std::env::var("DS").unwrap();

  fs::create_dir_all(format!("dataset/{ds}/out/full")).unwrap();

  let vecs = read_vectors("base.fvecs", LittleEndian::read_f32_into);

  let full_dists = vecs
    .par_iter()
    .flat_map(|vec_i| {
      vecs
        .par_iter()
        .map(|vec_j| metric_euclidean(&vec_i.view(), &vec_j.view()))
    })
    .collect::<Vec<_>>();
  println!("Calculated full dists");
  File::create(format!("dataset/{ds}/out/full/edge_dists.mat"))
    .unwrap()
    .write_all(
      &full_dists
        .into_iter()
        .flat_map(|d| (d as f32).to_le_bytes())
        .collect_vec(),
    )
    .unwrap();
  println!("Exported full dists");
}
