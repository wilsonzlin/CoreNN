use crate::new_pb;
use ahash::HashSet;
use bytemuck::cast_slice;
use clap::Args;
use half::f16;
use libroxanne::cfg::RoxanneDbCfg;
use libroxanne::util::AsyncConcurrentIteratorExt;
use libroxanne::RoxanneDb;
use libroxanne::RoxanneDbDir;
use ndarray::Array1;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::fs::read;
use tokio::fs::read_to_string;
use tokio::fs::File;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;

#[derive(Args)]
pub struct EvalArgs {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Path to packed f16 matrix where each row is a query vector.
  #[arg(long)]
  queries: PathBuf,

  /// Path to ND-JSON, where each line is an array of strings representing ID of nearest neighbors for its corresponding query.
  #[arg(long)]
  knn: PathBuf,
}

impl EvalArgs {
  pub async fn exec(self) {
    let dir = RoxanneDbDir::new(self.path);
    let idx = Arc::new(RoxanneDb::<f16, f32>::open(dir.db()).await);
    let cfg_raw = read_to_string(dir.cfg()).await.unwrap();
    let cfg: RoxanneDbCfg = toml::from_str(&cfg_raw).unwrap();
    tracing::info!("loaded database");

    let queries: Vec<Array1<f16>> = {
      let raw = read(&self.queries).await.unwrap();
      let raw_vec_len = size_of::<f16>() * cfg.dim;
      assert_eq!(raw.len() % raw_vec_len, 0);
      raw
        .chunks(raw_vec_len)
        .map(|raw_vec| Array1::from_vec(cast_slice(raw_vec).to_vec()))
        .collect()
    };
    tracing::info!("loaded query IDs");
    let knn_ids = {
      let f = File::open(&self.knn).await.unwrap();
      let rd = BufReader::new(f);
      let mut lines = rd.lines();
      let mut knns = Vec::new();
      while let Some(l) = lines.next_line().await.unwrap() {
        let knn: HashSet<String> = serde_json::from_str(&l).unwrap();
        knns.push(knn);
      }
      knns
    };
    tracing::info!("loaded k-NN");

    let pb = new_pb(queries.len());
    let correct = Arc::new(AtomicUsize::new(0));
    let total = Arc::new(AtomicUsize::new(0));
    zip(queries, knn_ids)
      .map(|(q, k)| {
        (
          idx.clone(),
          pb.clone(),
          correct.clone(),
          total.clone(),
          q,
          k,
        )
      })
      .spawn_for_each(
        |(idx, pb, correct, total, query, knn_expected)| async move {
          let k = knn_expected.len();
          let knn_got = idx
            .query(&query.view(), k)
            .await
            .into_iter()
            .map(|e| e.0)
            .collect::<HashSet<_>>();

          correct.fetch_add(
            knn_expected.intersection(&knn_got).count(),
            Ordering::Relaxed,
          );
          total.fetch_add(k, Ordering::Relaxed);

          let correct = correct.load(Ordering::Relaxed);
          let total = total.load(Ordering::Relaxed);
          let pc = correct as f64 / total as f64 * 100.0;
          pb.set_message(format!("Correct: {pc:.2}% ({correct}/{total})"));
          pb.inc(1);
        },
      )
      .await;
    pb.finish();
    tracing::info!("all done!");
  }
}
