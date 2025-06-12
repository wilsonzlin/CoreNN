use super::Compressor;
use super::CV;
use crate::metric::StdMetric;
use crate::vec::VecData;
use std::sync::Arc;

#[derive(Debug)]
pub struct TruncCompressor {
  dim: usize,
}

impl TruncCompressor {
  pub fn new(dim: usize) -> Self {
    Self { dim }
  }
}

impl Compressor for TruncCompressor {
  fn into_compressed(&self, v: VecData) -> CV {
    macro_rules! trunc {
      ($a:expr) => {{
        $a.truncate(self.dim);
        $a
      }};
    }
    let v = match v {
      VecData::BF16(mut a) => VecData::BF16(trunc!(a)),
      VecData::F16(mut a) => VecData::F16(trunc!(a)),
      VecData::F32(mut a) => VecData::F32(trunc!(a)),
      VecData::F64(mut a) => VecData::F64(trunc!(a)),
    };
    Arc::new(v)
  }

  fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f64 {
    let a = a.downcast_ref::<VecData>().unwrap();
    let b = b.downcast_ref::<VecData>().unwrap();
    let f = metric.get_fn();
    f(a, b)
  }
}
