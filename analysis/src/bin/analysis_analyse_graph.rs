use ahash::HashMap;
use clap::Parser;
use libroxanne::common::Id;
use libroxanne::common::Metric;
use libroxanne::common::PrecomputedDists;
use libroxanne::pq::ProductQuantizer;
use libroxanne::search::GreedySearchable;
use libroxanne::search::GreedySearchableSync;
use libroxanne::search::Points;
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

  /// This must be at least 100 (as k=100). For consistent benchmarking results, this should be 100.
  #[arg(long, default_value_t = 100)]
  search_list_cap: usize,

  #[arg(long)]
  subspaces: Option<usize>,
}

struct EvalGraph {
  adj_list: HashMap<Id, Vec<Id>>,
  vectors: Array2<f32>,
  pq: Option<ProductQuantizer<f32>>,
  metric: Metric<f32>,
  vectors_pq: Option<Array2<u8>>,
  medoid: Id,
}

impl Points<f32, f32> for EvalGraph {
  type Point<'a> = Array1<f32>;

  fn metric(&self) -> Metric<f32> {
    self.metric
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }

  fn get_point(&self, id: Id) -> Array1<f32> {
    match self.vectors_pq.as_ref() {
      Some(vecs) => self.pq.as_ref().unwrap().decode_1(&vecs.row(id)),
      None => self.vectors.row(id).to_owned(),
    }
  }
}

impl GreedySearchable<f32, f32> for EvalGraph {
  type FullVec = Array1<f32>;
  type Neighbors<'a> = Vec<Id>;

  fn medoid(&self) -> Id {
    self.medoid
  }
}

impl<'a> GreedySearchableSync<f32, f32> for EvalGraph {
  fn get_out_neighbors_sync(&self, id: Id) -> (Vec<Id>, Option<Array1<f32>>) {
    let neighbors = self.adj_list.get(&id).unwrap().clone();
    let full_vec = self.pq.as_ref().map(|_| self.vectors.row(id).to_owned());
    (neighbors, full_vec)
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

  let index = EvalGraph {
    adj_list: graph,
    medoid,
    metric: ds.info.metric.get_fn(),
    pq,
    vectors_pq,
    vectors: vecs,
  };

  let e = eval(
    &index,
    &qs.view(),
    &knns.view(),
    args.search_list_cap,
    args.beam_width,
  );
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
