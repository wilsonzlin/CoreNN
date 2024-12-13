use crate::eval;
use crate::Dataset;
use ahash::HashMap;
use libroxanne::common::Id;
use libroxanne::common::Metric;
use libroxanne::common::PrecomputedDists;
use libroxanne::pq::ProductQuantizer;
use libroxanne::search::GreedySearchable;
use libroxanne::search::GreedySearchableSync;
use libroxanne::search::Points;
use ndarray::Array1;
use ndarray::Array2;
use rmp_serde::from_slice;
use rmp_serde::to_vec_named;
use std::fs::exists;
use std::fs::read;
use std::fs::read_to_string;
use std::fs::write;

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

pub fn analyze_graph(
  ds: &Dataset,
  variant: &str,
  beam_width: usize,
  search_list_cap: usize,
  subspaces: Option<usize>,
) {
  let vecs = ds.read_vectors();
  let qs = ds.read_queries();
  let knns = ds.read_results();

  let out_dir = format!("dataset/{}/out/{}", ds.name, variant);

  let graph: HashMap<Id, Vec<Id>> =
    from_slice(&read(format!("{out_dir}/graph.msgpack")).unwrap()).unwrap();
  let medoid: Id = read_to_string(format!("{out_dir}/medoid.txt",))
    .unwrap()
    .parse()
    .unwrap();
  println!("Loaded graph");

  let (pq, vectors_pq) = match subspaces {
    Some(s) => {
      let path = format!("{out_dir}/pq-{s}.msgpack",);
      let pq = if exists(&path).unwrap() {
        let raw = read(path).unwrap();
        from_slice(&raw).unwrap()
      } else {
        let pq = ProductQuantizer::train(&vecs.view(), s);
        println!("Trained PQ");
        write(path, to_vec_named(&pq).unwrap()).unwrap();
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
    search_list_cap,
    beam_width,
  );
  println!("Evaluated all queries");
  write(
    format!(
      "{out_dir}/query_metrics{}.msgpack",
      subspaces.map(|s| format!("_pq{s}")).unwrap_or_default()
    ),
    to_vec_named(&e.query_metrics).unwrap(),
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
