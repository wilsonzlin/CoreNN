use bytemuck::cast_slice;
use clap::Args;
use futures::StreamExt;
use libroxanne::cfg::RoxanneDbCfg;
use libroxanne::db::Db;
use libroxanne::RoxanneDbDir;
use std::path::PathBuf;
use tokio::fs::create_dir_all;
use tokio::fs::read_to_string;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::io::BufWriter;

#[derive(Args)]
pub struct ExportVectorsArgs {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Output directory to write Roxanne index to.
  #[arg(long)]
  out: PathBuf,
}

impl ExportVectorsArgs {
  pub async fn exec(self) {
    let dir = RoxanneDbDir::new(self.path);

    let cfg_raw = read_to_string(&dir.cfg()).await.unwrap();
    let cfg: RoxanneDbCfg = toml::from_str(&cfg_raw).unwrap();

    let db = Db::open(dir.db()).await;

    let dim = cfg.dim;

    create_dir_all(&self.out).await.unwrap();

    let out_ids = File::create(self.out.join("ids.bin")).await.unwrap();
    let mut out_ids = BufWriter::new(out_ids);
    let out_vecs = File::create(self.out.join("vecs.bin")).await.unwrap();
    let mut out_vecs = BufWriter::new(out_vecs);

    let mut stream = db.iter_nodes::<half::f16>();
    while let Some((id, node)) = stream.next().await {
      out_ids.write_u64_le(id.try_into().unwrap()).await.unwrap();
      assert_eq!(node.vector.len(), dim);
      out_vecs.write_all(cast_slice(&node.vector)).await.unwrap();
    }

    out_ids.flush().await.unwrap();
    out_vecs.flush().await.unwrap();

    tracing::info!("all done!");
  }
}
