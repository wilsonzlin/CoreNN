use bytemuck::Pod;
use corenn_kernels::Kernel;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Dtype {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  QI8 = 3,
}

pub trait Scalar: Kernel + Pod {
  const DTYPE: Dtype;
  fn to_f32(self) -> f32;
  fn from_f32(v: f32) -> Self;
}

impl Scalar for f32 {
  const DTYPE: Dtype = Dtype::F32;

  #[inline]
  fn to_f32(self) -> f32 {
    self
  }

  #[inline]
  fn from_f32(v: f32) -> Self {
    v
  }
}

impl Scalar for half::f16 {
  const DTYPE: Dtype = Dtype::F16;

  #[inline]
  fn to_f32(self) -> f32 {
    self.to_f32()
  }

  #[inline]
  fn from_f32(v: f32) -> Self {
    Self::from_f32(v)
  }
}

impl Scalar for half::bf16 {
  const DTYPE: Dtype = Dtype::BF16;

  #[inline]
  fn to_f32(self) -> f32 {
    self.to_f32()
  }

  #[inline]
  fn from_f32(v: f32) -> Self {
    Self::from_f32(v)
  }
}
