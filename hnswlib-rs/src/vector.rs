use crate::scalar::Dtype;
use crate::scalar::Scalar;
use std::marker::PhantomData;

pub trait VectorRef: Copy {
  fn len(self) -> usize;
}

impl<T> VectorRef for &[T] {
  #[inline]
  fn len(self) -> usize {
    self.len()
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Qi8Ref<'a> {
  pub data: &'a [i8],
  pub scale: f32,
  pub zero_point: i8,
}

impl VectorRef for Qi8Ref<'_> {
  #[inline]
  fn len(self) -> usize {
    self.data.len()
  }
}

pub trait VectorFamily: Copy + Clone + Send + Sync + 'static {
  type Ref<'a>: VectorRef
  where
    Self: 'a;

  const DTYPE: Dtype;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Dense<S: Scalar>(PhantomData<S>);

impl<S: Scalar> VectorFamily for Dense<S> {
  type Ref<'a>
    = &'a [S]
  where
    Self: 'a;

  const DTYPE: Dtype = S::DTYPE;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Qi8;

impl VectorFamily for Qi8 {
  type Ref<'a>
    = Qi8Ref<'a>
  where
    Self: 'a;

  const DTYPE: Dtype = Dtype::QI8;
}

pub trait VectorView<F: VectorFamily>: Clone + Send + Sync {
  fn view<'a>(&'a self) -> <F as VectorFamily>::Ref<'a>;
}

impl<S: Scalar> VectorView<Dense<S>> for &[S] {
  fn view<'a>(&'a self) -> &'a [S] {
    *self
  }
}

impl<'v> VectorView<Qi8> for Qi8Ref<'v> {
  fn view<'a>(&'a self) -> Qi8Ref<'a> {
    Qi8Ref {
      data: self.data,
      scale: self.scale,
      zero_point: self.zero_point,
    }
  }
}
