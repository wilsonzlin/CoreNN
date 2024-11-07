use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use roxanne_analysis::analyse_index;
use roxanne_analysis::read_vectors;
use roxanne_analysis::read_vectors_dims;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long, default_value_t = 16)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.33)]
  search_list_cap_mul: f64,
}

fn main() {
  let ds = std::env::var("DS").unwrap();

  let args = Args::parse();

  fs::create_dir_all(format!("dataset/{ds}/out/random")).unwrap();

  let vecs = read_vectors("base.fvecs", LittleEndian::read_f32_into);
  let k = read_vectors_dims("groundtruth.ivecs");

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: 1.1,
    query_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
    update_batch_size: 64,
    update_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };

  let index = InMemoryVamana::init_random_index(
    vecs.into_iter().enumerate().collect(),
    metric_euclidean,
    params,
    10_000,
  );
  println!("Built graph");

  analyse_index("random", &index);
}
