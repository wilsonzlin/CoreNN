use crate::vec::VecData;
use corenn_kernels::cosine_distance;
use corenn_kernels::cosine_distance_qi8;
use corenn_kernels::inner_product_distance;
use corenn_kernels::inner_product_distance_qi8;
use corenn_kernels::Kernel;
use corenn_kernels::l2_sq_qi8;

pub fn dist_l2_sq(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => <half::bf16 as Kernel>::l2_sq(a, b),
    (VecData::F16(a), VecData::F16(b)) => <half::f16 as Kernel>::l2_sq(a, b),
    (VecData::F32(a), VecData::F32(b)) => <f32 as Kernel>::l2_sq(a, b),
    (VecData::QI8(a), VecData::QI8(b)) => l2_sq_qi8(&a.data, a.scale, a.zero_point, &b.data, b.scale, b.zero_point),
    _ => panic!("differing dtypes"),
  }
}

pub fn dist_inner_product(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => inner_product_distance::<half::bf16>(a, b),
    (VecData::F16(a), VecData::F16(b)) => inner_product_distance::<half::f16>(a, b),
    (VecData::F32(a), VecData::F32(b)) => inner_product_distance::<f32>(a, b),
    (VecData::QI8(a), VecData::QI8(b)) => inner_product_distance_qi8(&a.data, a.scale, a.zero_point, &b.data, b.scale, b.zero_point),
    _ => panic!("differing dtypes"),
  }
}

pub fn dist_cosine(a: &VecData, b: &VecData) -> f32 {
  assert_eq!(a.dim(), b.dim());
  match (a, b) {
    (VecData::BF16(a), VecData::BF16(b)) => cosine_distance::<half::bf16>(a, b),
    (VecData::F16(a), VecData::F16(b)) => cosine_distance::<half::f16>(a, b),
    (VecData::F32(a), VecData::F32(b)) => cosine_distance::<f32>(a, b),
    (VecData::QI8(a), VecData::QI8(b)) => cosine_distance_qi8(&a.data, a.scale, a.zero_point, &b.data, b.scale, b.zero_point),
    _ => panic!("differing dtypes"),
  }
}
