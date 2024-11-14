use ahash::HashMap;
use clap::Parser;
use libroxanne::pq::ProductQuantizer;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use ndarray::Array1;
use ndarray::Array2;
use roxanne_analysis::eval;
use roxanne_analysis::Dataset;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg()]
  variant: String,

  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long)]
  search_list_cap: usize,

  #[arg(long)]
  subspaces: Option<usize>,
}

struct EvalGraph {
  adj_list: HashMap<Id, Vec<Id>>,
  vectors: Array2<f32>,
  pq: Option<ProductQuantizer<f32>>,
  vectors_pq: Option<Array2<u8>>,
}

impl GreedySearchable<f32> for EvalGraph {
  fn get_point(&self, id: Id) -> Array1<f32> {
    match self.vectors_pq.as_ref() {
      Some(vecs) => self.pq.as_ref().unwrap().decode_1(&vecs.row(id)),
      None => self.vectors.row(id).to_owned(),
    }
  }

  fn get_out_neighbors(&self, id: Id) -> (Vec<Id>, Option<Array1<f32>>) {
    let neighbors = self.adj_list.get(&id).unwrap().clone();
    let full_vec = self.pq.as_ref().map(|_| self.vectors.row(id).to_owned());
    (neighbors, full_vec)
  }
}

impl VamanaDatastore<f32> for EvalGraph {
  fn set_point(&self, _id: Id, _point: Array1<f32>) {
    panic!("read only");
  }

  fn set_out_neighbors(&self, _id: Id, _neighbors: Vec<Id>) {
    panic!("read only");
  }
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();
  let out_dir = args.variant;

  let vecs = ds.read_vectors();
  let qs = ds.read_queries();
  let knns = ds.read_results();

  let graph: HashMap<Id, Vec<Id>> = rmp_serde::from_slice(
    &fs::read(format!("dataset/{}/out/{out_dir}/graph.msgpack", ds.name)).unwrap(),
  )
  .unwrap();
  let medoid: Id = fs::read_to_string(format!("dataset/{}/out/{out_dir}/medoid.txt", ds.name))
    .unwrap()
    .parse()
    .unwrap();
  println!("Loaded graph");

  let (pq, vectors_pq) = match args.subspaces {
    Some(s) => {
      let path = format!("dataset/{}/out/{out_dir}/pq-{s}.msgpack", ds.name);
      let pq = if fs::exists(&path).unwrap() {
        let raw = fs::read(path).unwrap();
        rmp_serde::from_slice(&raw).unwrap()
      } else {
        let pq = ProductQuantizer::train(&vecs.view(), s);
        println!("Trained PQ");
        fs::write(path, rmp_serde::to_vec_named(&pq).unwrap()).unwrap();
        pq
      };
      let vecs_pq = pq.encode(&vecs.view());
      println!("Calculated PQ vectors");
      (Some(pq), Some(vecs_pq))
    }
    None => (None, None),
  };

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: 1,         // Irrelevant.
    distance_threshold: 1.0, // Irrelevant.
    query_search_list_cap: args.search_list_cap,
    update_batch_size: 1,      // Irrelevant.
    update_search_list_cap: 1, // Irrelevant.
  };

  let index = EvalGraph {
    adj_list: graph,
    pq,
    vectors_pq,
    vectors: vecs,
  };
  let index = Vamana::new(index, metric_euclidean, medoid, params);

  let e = eval(&index, &qs.view(), &knns.view());
  println!("Evaluated all queries");
  fs::write(
    format!(
      "dataset/{}/out/{out_dir}/query_metrics{}.msgpack",
      ds.name,
      args
        .subspaces
        .map(|s| format!("_pq{s}"))
        .unwrap_or_default()
    ),
    rmp_serde::to_vec_named(&e.query_metrics).unwrap(),
  )
  .unwrap();
  println!("Exported query metrics");

  println!(
    "Correct: {:.2}% ({}/{})",
    e.ratio() * 100.0,
    e.correct,
    e.total,
  );
}
