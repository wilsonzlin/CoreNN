use ahash::HashMap;
use ahash::HashMapExt;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use dashmap::DashMap;
use itertools::Itertools;
use libroxanne::pq::ProductQuantizer;
use libroxanne::vamana::calc_approx_medoid;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use ndarray::stack;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use roxanne_analysis::analyse_index;
use roxanne_analysis::eval;
use roxanne_analysis::read_vectors;
use roxanne_analysis::read_vectors_dims;
use std::fs;
use std::fs::File;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long, default_value_t = 1.33)]
  search_list_cap_mul: f64,

  #[arg(long, default_value_t = 32)]
  subspaces: usize,
}

struct PQGraph {
  adj_list: DashMap<Id, Vec<Id>>,
  id_to_point: DashMap<Id, Array1<f32>>,
  pq: ProductQuantizer<f32>,
  id_to_pq_point: DashMap<Id, Array1<u8>>,
}

impl GreedySearchable<f32> for PQGraph {
  fn get_point(&self, id: Id) -> Array1<f32> {
    let pq = self.id_to_pq_point.get(&id).unwrap();
    self.pq.decode_1(&pq.view())
  }

  fn get_out_neighbors(&self, id: Id) -> (Vec<Id>, Option<Array1<f32>>) {
    let neighbors = self.adj_list.get(&id).unwrap().clone();
    let full_emb = self.id_to_point.get(&id).unwrap().clone();
    (neighbors, Some(full_emb))
  }
}

impl VamanaDatastore<f32> for PQGraph {
  fn set_point(&self, _id: Id, _point: Array1<f32>) {
    panic!("read only");
  }

  fn set_out_neighbors(&self, _id: Id, _neighbors: Vec<Id>) {
    panic!("read only");
  }
}

fn main() {
  let ds = std::env::var("DS").unwrap();

  let args = Args::parse();

  let vecs = read_vectors("base.fvecs", LittleEndian::read_f32_into);
  let queries = read_vectors("query.fvecs", LittleEndian::read_f32_into);
  let ground_truth = read_vectors("groundtruth.ivecs", LittleEndian::read_u32_into);
  let n = vecs.len();
  let k = ground_truth[0].dim();

  let medoid: Id = {
    let path = format!("dataset/{ds}/out/vamana/vamana.medoid.txt");
    let raw = std::fs::read(path).unwrap();
    let s = String::from_utf8(raw).unwrap();
    s.parse().unwrap()
  };
  let graph: DashMap<Id, Vec<Id>> = rmp_serde::from_slice(
    &std::fs::read(format!("dataset/{ds}/out/vamana/graph.msgpack")).unwrap(),
  )
  .unwrap();
  println!("Loaded graph");

  let mat = stack(Axis(0), &vecs.iter().map(|v| v.view()).collect_vec()).unwrap();
  let pq = ProductQuantizer::train(&mat.view(), args.subspaces);
  println!("Trained PQ");

  let mat_pq = pq.encode(&mat.view());
  let id_to_pq_point = (0..n)
    .map(|id| (id, mat_pq.row(id).to_owned()))
    .collect::<DashMap<_, _>>();
  println!("Calculated PQ vectors");

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: graph.iter().map(|e| e.len()).max().unwrap(),
    distance_threshold: 1.1,
    query_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
    update_batch_size: num_cpus::get(),
    update_search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };
  println!("{params:?}");
  let id_to_point = vecs.into_iter().enumerate().collect();
  let ds = PQGraph {
    adj_list: graph,
    id_to_point,
    id_to_pq_point,
    pq,
  };
  let index = Vamana::new(ds, metric_euclidean, medoid, params);
  println!("Loaded graph");

  let e = eval(&index, &queries, &ground_truth);
  println!(
    "Correct: {:.2}% ({}/{})",
    e.ratio() * 100.0,
    e.correct,
    e.total,
  );
}
