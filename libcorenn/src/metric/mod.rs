use crate::metric::l2::dist_l2;
use crate::vec::VecData;
use cosine::dist_cosine;
use serde::Deserialize;
use serde::Serialize;
use strum_macros::Display;
use strum_macros::EnumString;

pub mod cosine;
pub mod l2;

pub type Metric = fn(&VecData, &VecData) -> f64;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, EnumString, Serialize, Deserialize)]
pub enum StdMetric {
  L2,
  Cosine,
}

impl StdMetric {
  pub fn get_fn(self) -> Metric {
    match self {
      StdMetric::L2 => dist_l2,
      StdMetric::Cosine => dist_cosine,
    }
  }
}
