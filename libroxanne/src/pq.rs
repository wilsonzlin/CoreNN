use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use ndarray::s;
use ndarray::Array2;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize)]
pub struct ProductQuantizer {
  dims: usize,
  subspace_codebooks: Vec<KMeans<f32, L2Dist>>,
}

impl ProductQuantizer {
  pub fn train(mat: Array2<f32>, subspaces: usize) -> Self {
    let dims = mat.shape()[1];
    assert_eq!(dims % subspaces, 0);
    let subdims = dims / subspaces;
    let subspace_codebooks = (0..subspaces)
      .into_par_iter()
      .map(|i| {
        let submat = mat.slice(s![.., i * subdims..(i + 1) * subdims]);
        let obs = DatasetBase::new(submat.to_owned(), ());
        KMeans::params(256).fit(&obs).unwrap()
      })
      .collect::<Vec<_>>();
    ProductQuantizer {
      dims,
      subspace_codebooks,
    }
  }

  pub fn encode(&self, mat: Array2<f32>) -> Array2<u8> {
    assert_eq!(mat.shape()[1], self.dims);
    let n = mat.shape()[0];
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    let mut codes = Array2::zeros((n, subspaces));
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let submat = mat.slice(s![.., i * subdims..(i + 1) * subdims]);
      let obs = DatasetBase::new(submat.to_owned(), ());
      let labels = codebook.predict(&obs);
      assert_eq!(labels.shape(), &[n]);
      codes
        .slice_mut(s![.., i])
        .assign(&labels.mapv(|x| u8::try_from(x).unwrap()));
    }
    codes
  }

  pub fn decode(&self, codes: &Array2<u8>) -> Array2<f32> {
    let n = codes.shape()[0];
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    assert_eq!(codes.shape()[1], subspaces);
    let mut mat = Array2::<f32>::zeros((n, self.dims));
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let dec = codes
        .column(i)
        .mapv(|idx| codebook.centroids().row(idx.into()).to_owned());
      for j in 0..n {
        mat
          .slice_mut(s![j, i * subdims..(i + 1) * subdims])
          .assign(&dec[j]);
      }
    }
    mat
  }
}

#[cfg(test)]
mod tests {
  use super::ProductQuantizer;
  use croaring::Bitmap;
  use itertools::Itertools;
  use ndarray::Array;
  use ndarray::Array2;
  use ndarray::ArrayView1;
  use ndarray::Axis;
  use num_traits::Float;
  use num_traits::One;
  use num_traits::Zero;
  use ordered_float::OrderedFloat;
  use rand::distributions::Uniform;
  use rand::prelude::Distribution;
  use rand::thread_rng;
  use std::iter::zip;
  use std::ops::AddAssign;
  use std::ops::Sub;

  fn pairwise_euclidean_distance<T>(matrix: &Array2<T>) -> Array2<T>
  where
    T: Float + Zero + One + Sub<Output = T> + AddAssign + std::fmt::Debug,
  {
    let n = matrix.nrows();
    let mut distances = Array2::zeros((n, n));

    // Compute squared Euclidean distances
    for i in 0..n {
      for j in i + 1..n {
        let squared_dist = squared_euclidean_distance(matrix.row(i), matrix.row(j));
        distances[[i, j]] = squared_dist;
        distances[[j, i]] = squared_dist; // Distance matrix is symmetric
      }
    }

    // Take square root to get Euclidean distances
    distances.mapv_inplace(T::sqrt);

    distances
  }

  fn squared_euclidean_distance<T>(a: ArrayView1<T>, b: ArrayView1<T>) -> T
  where
    T: Float + Zero + AddAssign,
  {
    a.iter()
      .zip(b.iter())
      .map(|(&x, &y)| (x - y) * (x - y))
      .fold(T::zero(), |acc, x| acc + x)
  }

  fn top_k_per_row(matrix: &Array2<f32>, k: usize) -> Vec<Bitmap> {
    matrix
      .axis_iter(Axis(0))
      .map(|row| {
        row
          .iter()
          .enumerate()
          .sorted_by_key(|(_, &x)| OrderedFloat(x))
          .take(k)
          .map(|(i, _)| i.try_into().unwrap())
          .collect::<Bitmap>()
      })
      .collect()
  }

  #[test]
  fn test_pq() {
    let mut rng = thread_rng();
    let dist = Uniform::new(-10.0f32, 10.0f32);
    let n = 1000;
    let mat = Array::from_shape_fn((n, 128), |_| dist.sample(&mut rng));
    println!("Generated {:?}", mat.shape());

    let pq = ProductQuantizer::train(mat.clone(), 64);
    println!("Trained");
    let mat_pq = pq.encode(mat.clone());
    println!("Quantized to {:?}", mat_pq.shape());

    let k = 10;
    let dists = pairwise_euclidean_distance(&mat);
    let dists_pq = pairwise_euclidean_distance(&pq.decode(&mat_pq));
    let mut correct = 0;
    for (top, top_pq) in zip(top_k_per_row(&dists, k), top_k_per_row(&dists_pq, k)) {
      correct += top.and_cardinality(&top_pq);
    }
    println!(
      "[PQ] Correct: {}/{} ({:.2}%)",
      correct,
      k * n,
      correct as f64 / (k * n) as f64 * 100.0
    );
  }
}
