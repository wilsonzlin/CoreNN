use crate::vec::VecData;
use corenn_kernels::cosine_distance;
use corenn_kernels::inner_product_distance;
use corenn_kernels::Kernel;

pub fn dist_l2_sq(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => <half::bf16 as Kernel>::l2_sq(a, b),
    (VecData::F16(a), VecData::F16(b)) => <half::f16 as Kernel>::l2_sq(a, b),
    (VecData::F32(a), VecData::F32(b)) => <f32 as Kernel>::l2_sq(a, b),
    _ => panic!("differing dtypes"),
  }
}

pub fn dist_inner_product(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => inner_product_distance::<half::bf16>(a, b),
    (VecData::F16(a), VecData::F16(b)) => inner_product_distance::<half::f16>(a, b),
    (VecData::F32(a), VecData::F32(b)) => inner_product_distance::<f32>(a, b),
    _ => panic!("differing dtypes"),
  }
}

pub fn dist_cosine(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => cosine_distance::<half::bf16>(a, b),
    (VecData::F16(a), VecData::F16(b)) => cosine_distance::<half::f16>(a, b),
    (VecData::F32(a), VecData::F32(b)) => cosine_distance::<f32>(a, b),
    _ => panic!("differing dtypes"),
  }
}
