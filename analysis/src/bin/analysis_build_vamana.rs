use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use roxanne_analysis::analyse_index;
use roxanne_analysis::Dataset;
use std::fs;
use std::sync::Arc;

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
  let ds = Dataset::init();

  let args = Args::parse();

  fs::create_dir_all(format!("dataset/{}/out/vamana", ds.name)).unwrap();

  let vecs = ds.read_vectors();
  println!("Loaded vectors");
  let n = ds.info.n;
  let k = ds.info.k;
  let precomputed_dists = ds.read_dists();
  println!("Loaded dists");

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    query_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
    update_batch_size: args.insert_batch_size,
    update_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };
  println!("Params: {params:?}");

  let index = InMemoryVamana::build_index(
    (0..n).map(|i| (i, vecs.row(i).to_owned())).collect(),
    metric_euclidean,
    params,
    10_000,
    Some(Arc::new((
      (0..vecs.len()).map(|i| (i, i)).collect(),
      precomputed_dists,
    ))),
  );
  fs::write(
    format!("dataset/{}/out/vamana/vamana.medoid.txt", ds.name),
    index.medoid().to_string(),
  )
  .unwrap();
  println!("Built graph");

  analyse_index(&ds, "vamana", &index);
}
