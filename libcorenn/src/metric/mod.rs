use crate::vec::VecData;
use serde::Deserialize;
use serde::Serialize;
use strum_macros::Display;
use strum_macros::EnumString;

pub mod ops;

pub type Metric = fn(&VecData, &VecData) -> f32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, EnumString, Serialize, Deserialize)]
pub enum StdMetric {
  L2Sq,
  Cosine,
  InnerProduct,
}

impl StdMetric {
  pub fn get_fn(self) -> Metric {
    match self {
      StdMetric::L2Sq => ops::dist_l2_sq,
      StdMetric::Cosine => ops::dist_cosine,
      StdMetric::InnerProduct => ops::dist_inner_product,
    }
  }
}
