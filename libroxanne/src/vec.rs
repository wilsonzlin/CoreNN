use bytemuck::cast_slice;
use derive_more::From;
use half::bf16;
use half::f16;
use ndarray::Array1;
use serde::Deserialize;
use serde::Serialize;

// We don't use ndarray because it doesn't support .trunc without copying.
// It's fine, as ArrayView can be created from a slice without copying.
#[derive(Clone, Debug, Deserialize, From, Serialize)]
pub enum VecData {
  BF16(Vec<bf16>),
  F16(Vec<f16>),
  F32(Vec<f32>),
  F64(Vec<f64>),
}

impl VecData {
  pub fn dim(&self) -> usize {
    match self {
      VecData::BF16(v) => v.len(),
      VecData::F16(v) => v.len(),
      VecData::F32(v) => v.len(),
      VecData::F64(v) => v.len(),
    }
  }

  /// Native endian.
  pub fn as_raw_bytes(&self) -> &[u8] {
    match self {
      VecData::BF16(v) => cast_slice(v),
      VecData::F16(v) => cast_slice(v),
      VecData::F32(v) => cast_slice(v),
      VecData::F64(v) => cast_slice(v),
    }
  }

  pub fn into_f32(self) -> Array1<f32> {
    let v = match self {
      VecData::BF16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
      VecData::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
      VecData::F32(v) => v,
      VecData::F64(v) => v.into_iter().map(|x| x as f32).collect(),
    };
    Array1::from_vec(v)
  }

  pub fn to_f32(&self) -> Array1<f32> {
    self.clone().into_f32()
  }
}
