use bytemuck::cast_slice;
use half::bf16;
use half::f16;
use ndarray::Array1;
use serde::Deserialize;
use serde::Serialize;

// We don't use ndarray because it doesn't support .trunc without copying.
// It's fine, as ArrayView can be created from a slice without copying.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum VecData {
  BF16(Vec<bf16>),
  F16(Vec<f16>),
  F32(Vec<f32>),
}

impl VecData {
  pub fn dim(&self) -> usize {
    match self {
      VecData::BF16(v) => v.len(),
      VecData::F16(v) => v.len(),
      VecData::F32(v) => v.len(),
    }
  }

  /// Native endian.
  pub fn as_raw_bytes(&self) -> &[u8] {
    match self {
      VecData::BF16(v) => cast_slice(v),
      VecData::F16(v) => cast_slice(v),
      VecData::F32(v) => cast_slice(v),
    }
  }

  pub fn into_f32(self) -> Array1<f32> {
    let v = match self {
      VecData::BF16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
      VecData::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
      VecData::F32(v) => v,
    };
    Array1::from_vec(v)
  }

  pub fn to_f32(&self) -> Array1<f32> {
    self.clone().into_f32()
  }
}

impl From<Vec<bf16>> for VecData {
  fn from(v: Vec<bf16>) -> Self {
    Self::BF16(v)
  }
}

impl From<Vec<f16>> for VecData {
  fn from(v: Vec<f16>) -> Self {
    Self::F16(v)
  }
}

impl From<Vec<f32>> for VecData {
  fn from(v: Vec<f32>) -> Self {
    Self::F32(v)
  }
}

// Intentionally no `From<Vec<f64>>`: require callers to explicitly cast.
