use super::Compressor;
use super::CV;
use crate::metric::StdMetric;
use crate::vec::VecData;
use crate::CoreNN;
use crate::Mode;
use itertools::Itertools;
use linfa::traits::FitWith;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa::Float;
use linfa_clustering::IncrKMeansError;
use linfa_clustering::KMeans;
use linfa_clustering::KMeansInit;
use linfa_nn::distance::L2Dist;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::cmp::min;
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize)]
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

  pub fn train_from_corenn(corenn: &CoreNN) -> ProductQuantizer<f32> {
    let samp_sz = min(corenn.cfg.pq_sample_size, corenn.count.get());
    // TODO Handle many gaps (i.e. deleted).
    let ids = (0..corenn.count.get()).choose_multiple(&mut thread_rng(), samp_sz);
    let Mode::Uncompressed(nodes) = &*corenn.mode.read() else {
      unreachable!();
    };
    let samp_nodes = nodes
      .multi_get(&ids)
      .into_iter()
      .filter_map(|n| n)
      .collect_vec();
    let actual_samp_sz = samp_nodes.len();

    let mut mat = Array2::zeros((actual_samp_sz, corenn.cfg.dim));
    for (i, node) in samp_nodes.into_iter().enumerate() {
      mat.row_mut(i).assign(&node.vector.to_f32());
    }
    let pq = ProductQuantizer::train(&mat.view(), corenn.cfg.pq_subspaces);

    tracing::info!(
      sample_inputs = actual_samp_sz,
      subspaces = corenn.cfg.pq_subspaces,
      "trained PQ"
    );

    pq
  }

  pub fn encode(&self, vec: &ArrayView1<T>) -> Vec<u8> {
    assert_eq!(vec.shape()[0], self.dims);
    let subspaces = self.subspace_codebooks.len();
    let subdims = self.dims / subspaces;
    let mut code = vec![0; subspaces];
    for (i, codebook) in self.subspace_codebooks.iter().enumerate() {
      let subvec = vec.slice(s![i * subdims..(i + 1) * subdims]);
      let obs = DatasetBase::new(subvec, ());
      let label = codebook.predict(&obs);
      code[i] = u8::try_from(label).unwrap();
    }
    code
  }
}

impl Compressor for ProductQuantizer<f32> {
  fn into_compressed(&self, v: VecData) -> CV {
    let v = v.into_f32();
    let view = ArrayView1::from(&v);
    Arc::new(self.encode(&view))
  }

  fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f64 {
    let a_codes = a.downcast_ref::<Vec<u8>>().unwrap();
    let b_codes = b.downcast_ref::<Vec<u8>>().unwrap();
    assert_eq!(a_codes.len(), b_codes.len());

    let num_subspaces = a_codes.len();

    match metric {
      StdMetric::L2 => {
        // Calculate L2 distance: sqrt(sum_i(||centroid_a_i - centroid_b_i||^2))
        // This is the sum of squared Euclidean distances between corresponding sub-centroids.
        let mut total_dist_sq_f64 = 0.0_f64;

        for i in 0..num_subspaces {
          let codebook = &self.subspace_codebooks[i]; // KMeans<f32, L2Dist>
          let centroids = codebook.centroids(); // &Array2<f32>
                                                // a_codes[i] and b_codes[i] are u8, representing centroid indices.
          let centroid_a_sub = centroids.row(a_codes[i].into()); // ArrayView1<f32>
          let centroid_b_sub = centroids.row(b_codes[i].into()); // ArrayView1<f32>

          let mut sub_dist_sq_f64 = 0.0_f64;
          // Assuming centroid_a_sub and centroid_b_sub have the same length (subdims)
          for k in 0..centroid_a_sub.len() {
            let diff_f32 = centroid_a_sub[k] - centroid_b_sub[k];
            let diff_f64: f64 = diff_f32.into(); // Promote to f64 for sum of squares
            sub_dist_sq_f64 += diff_f64 * diff_f64;
          }
          total_dist_sq_f64 += sub_dist_sq_f64;
        }
        total_dist_sq_f64.sqrt()
      }
      StdMetric::Cosine => {
        // Calculate Cosine distance: 1 - (A_hat . B_hat) / (||A_hat|| * ||B_hat||)
        // A_hat and B_hat are reconstructed vectors from centroids.
        // A_hat . B_hat = sum_i (centroid_a_sub_i . centroid_b_sub_i)
        // ||A_hat||^2 = sum_i ||centroid_a_sub_i||^2
        let mut total_dot_product_f64 = 0.0_f64;
        let mut total_norm_sq_a_f64 = 0.0_f64;
        let mut total_norm_sq_b_f64 = 0.0_f64;

        for i in 0..num_subspaces {
          let codebook = &self.subspace_codebooks[i];
          let centroids = codebook.centroids();
          let centroid_a_sub = centroids.row(a_codes[i] as usize);
          let centroid_b_sub = centroids.row(b_codes[i] as usize);

          // .dot() on ArrayView1<f32> returns f32. Cast to f64 for accumulation.
          total_dot_product_f64 += (centroid_a_sub.dot(&centroid_b_sub)) as f64;
          total_norm_sq_a_f64 += (centroid_a_sub.dot(&centroid_a_sub)) as f64; // ||ca_j||^2
          total_norm_sq_b_f64 += (centroid_b_sub.dot(&centroid_b_sub)) as f64; // ||cb_j||^2
        }

        // Handle cases with zero vectors using a small epsilon for squared norms.
        // A squared norm being less than EPSILON_SQ_NORM means the norm itself is very small.
        const EPSILON_SQ_NORM: f64 = 1e-12;

        if total_norm_sq_a_f64 < EPSILON_SQ_NORM && total_norm_sq_b_f64 < EPSILON_SQ_NORM {
          return 0.0; // Both reconstructed vectors are effectively zero. Cosine distance is 0.
        }
        if total_norm_sq_a_f64 < EPSILON_SQ_NORM || total_norm_sq_b_f64 < EPSILON_SQ_NORM {
          // One vector is effectively zero, the other is not. Cosine similarity is 0, distance is 1.
          return 1.0;
        }

        let norm_a_f64 = total_norm_sq_a_f64.sqrt();
        let norm_b_f64 = total_norm_sq_b_f64.sqrt();

        // It's highly unlikely norm_a_f64 or norm_b_f64 are zero if total_norm_sq_... were not,
        // but as a robust step, ensure denominator is not zero.
        // Using a slightly larger epsilon for norms themselves (sqrt of EPSILON_SQ_NORM).
        const EPSILON_NORM: f64 = 1e-6; // sqrt(1e-12)
        if norm_a_f64 < EPSILON_NORM || norm_b_f64 < EPSILON_NORM {
          return 1.0; // Denominator is effectively zero.
        }

        let cosine_similarity = total_dot_product_f64 / (norm_a_f64 * norm_b_f64);

        // Clamp cosine_similarity to [-1.0, 1.0] due to potential floating point inaccuracies.
        let clamped_similarity = cosine_similarity.max(-1.0).min(1.0);
        1.0 - clamped_similarity
      }
    }
  }
}
