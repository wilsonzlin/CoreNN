use clap::Args;
use libcorenn::store::schema::NODE;
use libcorenn::CoreNN;
use std::path::PathBuf;
use tokio::fs::create_dir_all;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::io::BufWriter;

#[derive(Args)]
pub struct ExportVectorsArgs {
  /// Path to a CoreNN DB.
  #[arg()]
  path: PathBuf,

  /// Output directory to write CoreNN index to.
  #[arg(long)]
  out: PathBuf,
}

impl ExportVectorsArgs {
  pub async fn exec(self) {
    let corenn = CoreNN::open(self.path);

    let dim = corenn.cfg().dim;

    create_dir_all(&self.out).await.unwrap();

    let out_ids = File::create(self.out.join("ids.bin")).await.unwrap();
    let mut out_ids = BufWriter::new(out_ids);
    let out_vecs = File::create(self.out.join("vecs.bin")).await.unwrap();
    let mut out_vecs = BufWriter::new(out_vecs);

    for (id, node) in NODE.iter(&corenn.internal_db()) {
      out_ids.write_u64_le(id.try_into().unwrap()).await.unwrap();
      assert_eq!(node.vector.dim(), dim);
      out_vecs
        .write_all(node.vector.as_raw_bytes())
        .await
        .unwrap();
    }

    out_ids.flush().await.unwrap();
    out_vecs.flush().await.unwrap();

    tracing::info!("all done!");
  }
}
