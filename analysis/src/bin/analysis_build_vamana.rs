use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use roxanne_analysis::analyse_index;
use roxanne_analysis::read_vectors;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long, default_value_t = 16)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.1)]
  distance_threshold: f64,

  #[arg(long, default_value_t = 64)]
  insert_batch_size: usize,

  #[arg(long, default_value_t = 1.33)]
  search_list_cap_mul: f64,
}

fn main() {
  let args = Args::parse();

  fs::create_dir_all("out/vamana").unwrap();

  let vecs = read_vectors("base.fvecs", LittleEndian::read_f32_into);
  let knns = read_vectors("groundtruth.ivecs", LittleEndian::read_u32_into);
  let k = knns[0].1.len();

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    query_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
    update_batch_size: args.insert_batch_size,
    update_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };
  println!("Params: {params:?}");

  let index = InMemoryVamana::build_index(vecs, metric_euclidean, params, 10_000, None);
  println!("Built graph");

  analyse_index("vamana", &index);
}
