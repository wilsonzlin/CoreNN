use super::Compressor;
use crate::util::AsyncConcurrentIteratorExt;
use crate::Roxanne;
use futures::StreamExt;
use half::f16;
use linfa::traits::FitWith;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa::Float;
use linfa_clustering::IncrKMeansError;
use linfa_clustering::KMeans;
use linfa_clustering::KMeansInit;
use linfa_nn::distance::L2Dist;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::cmp::min;

#[derive(Serialize, Deserialize)]
pub struct ProductQuantizer<T: Float> {
  dims: usize,
  subspace_codebooks: Vec<KMeans<T, L2Dist>>,
}

impl<T: Float> ProductQuantizer<T> {
  pub fn train(mat: &ArrayView2<T>, subspaces: usize) -> Self {
    // TODO Tune or allow config of this hyperparameter.
    let batch_size = 128;
    let dims = mat.shape()[1];
    assert_eq!(dims % subspaces, 0);
    let subdims = dims / subspaces;
    // Any NaN may completely break the K-means algorithm (it will fail to converge, due to distance and inertia calculations all failing to NaN).
    assert!(!mat.iter().any(|x| x.is_nan()));
    let subspace_codebooks = (0..subspaces)
      .into_par_iter()
      .map(|i| {
        let submat = mat.slice(s![.., i * subdims..(i + 1) * subdims]);
        // Mini-Batch K-means: https://docs.rs/linfa-clustering/latest/linfa_clustering/struct.KMeans.html#tutorial.
        // (It's much faster than standard K-means, while being similarly accurate.)
        // Shuffling is important to Mini-Batch K-means (see documentation above).
        let obs = DatasetBase::from(submat).shuffle(&mut thread_rng());
        // Use KMeansPara as it's faster for larger datasets. (See its comment.)
        // TODO Allow tuning `tolerance` and `n_runs`. `max_n_iterations` is not applicable to Mini-Batch K-means.
        let clf = KMeans::params(256).init_method(KMeansInit::KMeansPara);

        let mut cur: Option<KMeans<T, L2Dist>> = None;
        for batch in obs.sample_chunks(batch_size).cycle() {
          match clf.fit_with(cur, &batch) {
            // Early stop condition for the K-means loop.
            Ok(model) => {
              cur = Some(model);
              break;
            }
            // Continue running if not converged.
            Err(IncrKMeansError::NotConverged(model)) => {
              cur = Some(model);
            }
            Err(err) => {
              panic!("unexpected K-means error: {}", err);
            }
          };
        }
        cur.unwrap()
      })
      .collect::<Vec<_>>();
    ProductQuantizer {
      dims,
      subspace_codebooks,
    }
  }

  pub async fn train_from_roxanne(rox: &Roxanne) -> ProductQuantizer<f32> {
    // In-memory cache doesn't contain all vectors, and may be skewed.
    // Reservoir sampling from DB would require full disk read.
    // So we sample by ID, and assume there aren't many gaps (deletions).
    // TODO Handle gaps better.
    let samp_sz = min(rox.count.get(), rox.cfg.pq_sample_size());
    let ids = rand::seq::index::sample(&mut thread_rng(), rox.count.get(), samp_sz);
    let samp_nodes = ids
      .into_iter()
      .filter_map_concurrent(|id| rox.db.maybe_read_node(id))
      .collect::<Vec<_>>()
      .await;
    let actual_samp_sz = samp_nodes.len();

    let mut mat = Array2::zeros((actual_samp_sz, rox.dim.get()));
    for (i, node) in samp_nodes.into_iter().enumerate() {
      mat.row_mut(i).assign(&node.vector.mapv(|x| x.to_f32()));
    }
    let pq = ProductQuantizer::train(&mat.view(), rox.cfg.pq_subspaces());

    tracing::info!(
      sample_inputs = actual_samp_sz,
      subspaces = rox.cfg.pq_subspaces(),
      "trained PQ"
    );

    pq
  }

  pub fn encode(&self, mat: &ArrayView2<T>) -> Array2<u8> {
    assert_eq!(mat.shape()[1], self.dims);
    let n = mat.shape()[0];
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    let mut codes = Array2::zeros((n, subspaces));
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let submat = mat.slice(s![.., i * subdims..(i + 1) * subdims]);
      let obs = DatasetBase::new(submat, ());
      let labels = codebook.predict(&obs);
      assert_eq!(labels.shape(), &[n]);
      codes
        .slice_mut(s![.., i])
        .assign(&labels.mapv(|x| u8::try_from(x).unwrap()));
    }
    codes
  }

  pub fn encode_1(&self, vec: &ArrayView1<T>) -> Array1<u8> {
    assert_eq!(vec.shape()[0], self.dims);
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    let mut code = Array1::zeros(subspaces);
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let subvec = vec.slice(s![i * subdims..(i + 1) * subdims]);
      let obs = DatasetBase::new(subvec, ());
      let label = codebook.predict(&obs);
      code[i] = u8::try_from(label).unwrap();
    }
    code
  }

  pub fn decode(&self, codes: &ArrayView2<u8>) -> Array2<T> {
    let n = codes.shape()[0];
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    assert_eq!(codes.shape()[1], subspaces);
    let mut mat = Array2::<T>::zeros((n, self.dims));
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

  pub fn decode_1(&self, code: &ArrayView1<u8>) -> Array1<T> {
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    assert_eq!(code.dim(), subspaces);
    let mut mat = Array1::<T>::zeros(self.dims);
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let dec = codebook.centroids().row(code[i].into()).to_owned();
      mat
        .slice_mut(s![i * subdims..(i + 1) * subdims])
        .assign(&dec);
    }
    mat
  }
}

impl Compressor for ProductQuantizer<f32> {
  fn compress(&self, v: &ArrayView1<f16>) -> Vec<u8> {
    self.encode_1(&v.mapv(|x| x.to_f32()).view()).to_vec()
  }

  fn decompress(&self, v: &[u8]) -> Array1<f16> {
    let v = ArrayView1::from(v);
    self.decode_1(&v).mapv(|x| f16::from_f32(x))
  }
}

#[cfg(test)]
mod tests {
  use super::ProductQuantizer;
  use crate::common::dist_l2;
  use ahash::HashSet;
  use half::f16;
  use itertools::Itertools;
  use ndarray::Array;
  use ndarray::Array2;
  use ndarray::Axis;
  use ordered_float::OrderedFloat;
  use rand::distributions::Uniform;
  use rand::prelude::Distribution;
  use rand::thread_rng;
  use std::iter::zip;

  fn pairwise_euclidean_distance(matrix: &Array2<f16>) -> Array2<f32> {
    let n = matrix.nrows();
    let mut distances = Array2::zeros((n, n));

    // Compute squared Euclidean distances
    for i in 0..n {
      for j in i + 1..n {
        let squared_dist = dist_l2(&matrix.row(i), &matrix.row(j));
        distances[[i, j]] = squared_dist;
        distances[[j, i]] = squared_dist; // Distance matrix is symmetric
      }
    }

    // Take square root to get Euclidean distances
    distances.mapv_inplace(|x| x.sqrt());

    distances
  }

  fn top_k_per_row(matrix: &Array2<f32>, k: usize) -> Vec<HashSet<usize>> {
    matrix
      .axis_iter(Axis(0))
      .map(|row| {
        row
          .iter()
          .enumerate()
          .sorted_by_key(|(_, &x)| OrderedFloat(x))
          .take(k)
          .map(|(i, _)| i.try_into().unwrap())
          .collect::<HashSet<_>>()
      })
      .collect()
  }

  #[test]
  fn test_pq() {
    let mut rng = thread_rng();
    let dist = Uniform::new(-10.0f32, 10.0f32);
    let n = 1000;
    let mat = Array::from_shape_fn((n, 128), |_| dist.sample(&mut rng));

    let pq = ProductQuantizer::train(&mat.view(), 64);
    let mat_pq = pq.encode(&mat.view());

    let k = 10;
    let dists = pairwise_euclidean_distance(&mat.mapv(f16::from_f32));
    let dists_pq = pairwise_euclidean_distance(&pq.decode(&mat_pq.view()).mapv(f16::from_f32));
    let mut correct = 0;
    for (top, top_pq) in zip(top_k_per_row(&dists, k), top_k_per_row(&dists_pq, k)) {
      correct += top.intersection(&top_pq).count();
    }
    println!(
      "[PQ] Correct: {}/{} ({:.2}%)",
      correct,
      k * n,
      correct as f64 / (k * n) as f64 * 100.0
    );
  }
}
