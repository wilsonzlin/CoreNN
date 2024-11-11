use dashmap::DashMap;
use db::Db;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use ndarray::Array1;
use pq::ProductQuantizer;
use std::path::Path;
use vamana::Vamana;
use vamana::VamanaDatastore;

pub mod db;
pub mod pq;
pub mod queue;
pub mod vamana;

pub struct RoxanneDbReadOnly {
  db: Db,
  pq: ProductQuantizer<f32>,
  pq_vec_cache: DashMap<Id, Array1<u8>>,
}

impl GreedySearchable<f32> for RoxanneDbReadOnly {
  fn get_point(&self, id: Id) -> Array1<f32> {
    let vec = self.pq_vec_cache.entry(id).or_insert_with(|| {
      let raw = self.db.read_pq_vec(id);
      Array1::from_vec(raw)
    });
    self.pq.decode_1(&vec.view())
  }

  fn get_out_neighbors(&self, id: Id) -> (Vec<Id>, Option<Array1<f32>>) {
    let n = self.db.read_node(id);
    (n.neighbors, Some(Array1::from_vec(n.vector)))
  }
}

impl VamanaDatastore<f32> for RoxanneDbReadOnly {
  fn set_point(&self, _id: Id, _point: Array1<f32>) {
    panic!("read only");
  }

  fn set_out_neighbors(&self, _id: Id, _neighbors: Vec<Id>) {
    panic!("read only");
  }
}

impl RoxanneDbReadOnly {
  pub fn open(dir: impl AsRef<Path>) -> Vamana<f32, Self> {
    let db = Db::open(dir);
    let cfg = db.read_cfg();
    let medoid = db.read_medoid();
    let metric = db.read_metric().get_fn::<f32>();
    let pq = db.read_pq_model();
    Vamana::new(
      Self {
        db,
        pq,
        pq_vec_cache: DashMap::new(),
      },
      metric,
      medoid,
      cfg,
    )
  }

  pub fn raw_db(&self) -> &Db {
    &self.db
  }
}
