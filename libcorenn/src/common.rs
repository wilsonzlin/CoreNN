pub type Id = usize;

// Inspired by https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html.
pub fn nan_to_num<T: num::Float>(vec: &[T]) -> Vec<T> {
  vec
    .iter()
    .copied()
    .map(|v| {
      if v.is_nan() {
        T::zero()
      } else if v.is_infinite() {
        if v.is_sign_positive() {
          T::max_value()
        } else {
          // WARNING: This is not the same as f32::MIN_VALUE, which is 0.000...
          -T::max_value()
        }
      } else {
        v
      }
    })
    .collect()
}
