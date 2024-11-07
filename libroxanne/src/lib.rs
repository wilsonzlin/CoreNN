use db::DbReader;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use ndarray::Array1;
use std::path::Path;
use vamana::Vamana;
use vamana::VamanaDatastore;

pub mod db;
pub mod pq;
pub mod queue;
pub mod vamana;

pub struct RoxanneDbReadOnly {
  db: DbReader,
}

impl GreedySearchable<f32> for RoxanneDbReadOnly {
  fn get_point(&self, id: Id) -> Array1<f32> {
    Array1::from_vec(self.db.read_node(id).vector)
  }

  fn get_out_neighbors(&self, id: Id) -> Vec<Id> {
    self.db.read_node(id).neighbors
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
    let db = DbReader::new(dir);
    let cfg = db.read_cfg();
    let medoid = db.read_medoid();
    let metric = db.read_metric().get_fn::<f32>();
    Vamana::new(Self { db }, metric, medoid, cfg)
  }
}
