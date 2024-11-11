use byteorder::LittleEndian;
use byteorder::WriteBytesExt;
use clap::Parser;
use libroxanne::db::Db;
use libroxanne::db::NodeData;
use libroxanne_search::Id;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use std::fs::create_dir_all;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::str::from_utf8;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Output directory to write Roxanne index to.
  #[arg(long)]
  out: PathBuf,
}

fn main() {
  let args = Args::parse();

  let db = Db::open(&args.path);
  let dim = db.read_dim();

  create_dir_all(&args.out).unwrap();

  let out_ids = File::create(args.out.join("ids.bin")).unwrap();
  let mut out_ids = BufWriter::new(out_ids);
  let out_vecs = File::create(args.out.join("vecs.bin")).unwrap();
  let mut out_vecs = BufWriter::new(out_vecs);

  for e in db
    .inner()
    .full_iterator(IteratorMode::From(b"node/", Direction::Forward))
  {
    let (k, v) = e.unwrap();
    if !k.starts_with(b"node/") {
      break;
    };

    let id_raw = k.strip_prefix(b"node/").unwrap();
    let id = Id::from_str_radix(from_utf8(id_raw).unwrap(), 10).unwrap();
    out_ids
      .write_u64::<LittleEndian>(id.try_into().unwrap())
      .unwrap();

    let node = NodeData::deserialize(&v);
    assert_eq!(node.vector.len(), dim);
    for e in node.vector {
      out_vecs.write_f32::<LittleEndian>(e).unwrap();
    }
  }
  out_ids.flush().unwrap();
  out_vecs.flush().unwrap();

  println!("All done!");
}
