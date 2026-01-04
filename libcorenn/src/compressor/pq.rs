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
use corenn_kernels::Kernel;

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

  fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f32 {
    let a_codes = a.downcast_ref::<Vec<u8>>().unwrap();
    let b_codes = b.downcast_ref::<Vec<u8>>().unwrap();
    assert_eq!(a_codes.len(), b_codes.len());

    let num_subspaces = a_codes.len();

    match metric {
      StdMetric::L2Sq => {
        let mut total_dist_sq = 0.0_f32;

        for i in 0..num_subspaces {
          let codebook = &self.subspace_codebooks[i]; // KMeans<f32, L2Dist>
          let centroids = codebook.centroids(); // &Array2<f32>
                                                // a_codes[i] and b_codes[i] are u8, representing centroid indices.
          let centroid_a_sub = centroids.row(a_codes[i].into()); // ArrayView1<f32>
          let centroid_b_sub = centroids.row(b_codes[i].into()); // ArrayView1<f32>

          let mut sub_dist_sq = 0.0_f32;
          for k in 0..centroid_a_sub.len() {
            let diff = centroid_a_sub[k] - centroid_b_sub[k];
            sub_dist_sq += diff * diff;
          }
          total_dist_sq += sub_dist_sq;
        }
        total_dist_sq
      }
      StdMetric::Cosine | StdMetric::InnerProduct => {
        let mut total_dot = 0.0_f32;
        let mut total_norm_sq_a = 0.0_f32;
        let mut total_norm_sq_b = 0.0_f32;

        for i in 0..num_subspaces {
          let codebook = &self.subspace_codebooks[i];
          let centroids = codebook.centroids();
          let centroid_a_sub = centroids.row(a_codes[i] as usize);
          let centroid_b_sub = centroids.row(b_codes[i] as usize);

          let a_slice = centroid_a_sub.as_slice().unwrap();
          let b_slice = centroid_b_sub.as_slice().unwrap();
          let (dot, norm_sq_a, norm_sq_b) = <f32 as Kernel>::dot_and_norms(a_slice, b_slice);
          total_dot += dot;
          total_norm_sq_a += norm_sq_a;
          total_norm_sq_b += norm_sq_b;
        }

        match metric {
          StdMetric::InnerProduct => 1.0 - total_dot,
          StdMetric::Cosine => {
            const EPSILON_SQ_NORM: f32 = 1e-12;
            if total_norm_sq_a < EPSILON_SQ_NORM && total_norm_sq_b < EPSILON_SQ_NORM {
              return 0.0;
            }
            if total_norm_sq_a < EPSILON_SQ_NORM || total_norm_sq_b < EPSILON_SQ_NORM {
              return 1.0;
            }
            let denom = (total_norm_sq_a * total_norm_sq_b).sqrt();
            if denom == 0.0 {
              return 1.0;
            }
            let cosine_similarity = total_dot / denom;
            1.0 - cosine_similarity.clamp(-1.0, 1.0)
          }
          StdMetric::L2Sq => unreachable!(),
        }
      }
    }
  }
}
