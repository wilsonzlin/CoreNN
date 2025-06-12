use super::Store;
use super::WriteOp;
use rocksdb::BlockBasedOptions;
use rocksdb::Cache;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use rocksdb::Options;
use rocksdb::ReadOptions;
use rocksdb::WriteBatch;
use rocksdb::DB;
use std::path::Path;

pub fn rocksdb_options(create_if_missing: bool, error_if_exists: bool) -> Options {
  // https://github.com/facebook/rocksdb/wiki/Setup-Options-and-Basic-Tuning#other-general-options.
  let mut opt = Options::default();
  opt.create_if_missing(create_if_missing);
  opt.set_error_if_exists(error_if_exists);
  opt.set_max_background_jobs(num_cpus::get() as i32 * 2);
  opt.set_bytes_per_sync(1024 * 1024 * 4);
  opt.set_write_buffer_size(1024 * 1024 * 128);
  opt.set_compression_type(rocksdb::DBCompressionType::None);

  let cache = Cache::new_lru_cache(1024 * 1024 * 128);

  // https://github.com/facebook/rocksdb/wiki/Block-Cache.
  let mut bbt_opt = BlockBasedOptions::default();
  // Use close to a single page size to avoid excessive reads as that counters design of NodeData (seek DiskANN paper).
  bbt_opt.set_block_size(1024 * 4);
  bbt_opt.set_block_cache(&cache);
  bbt_opt.set_cache_index_and_filter_blocks(true);
  bbt_opt.set_pin_l0_filter_and_index_blocks_in_cache(true);
  bbt_opt.set_format_version(6);
  opt.set_block_based_table_factory(&bbt_opt);
  opt
}

pub struct RocksDBStore {
  db: DB,
}

impl RocksDBStore {
  pub fn open(path: impl AsRef<Path>, create_if_missing: bool, error_if_exists: bool) -> Self {
    let db = DB::open(&rocksdb_options(create_if_missing, error_if_exists), path).unwrap();
    Self { db }
  }
}

impl Store for RocksDBStore {
  fn get(&self, key: Vec<u8>) -> Option<Vec<u8>> {
    self.db.get(key).unwrap()
  }

  fn put(&self, key: Vec<u8>, value: Vec<u8>) {
    self.db.put(key, value).unwrap();
  }

  fn delete(&self, key: Vec<u8>) {
    self.db.delete(key).unwrap();
  }

  fn iter(
    &self,
    prefix: u8,
  ) -> Box<dyn Iterator<Item = (Box<[u8]>, Box<[u8]>)> + Send + Sync + '_> {
    // Optimize iterator for one-off table scan.
    let mut opts = ReadOptions::default();
    opts.set_pin_data(false);
    opts.fill_cache(false);
    opts.set_async_io(true);
    opts.set_auto_readahead_size(true);
    opts.set_iterate_lower_bound(&[prefix]);
    opts.set_iterate_upper_bound(&[prefix + 1]);
    opts.set_total_order_seek(true);
    Box::new(
      self
        .db
        .iterator_opt(IteratorMode::From(&[prefix], Direction::Forward), opts)
        .map(move |e| {
          let (k, v) = e.unwrap();
          assert_eq!(k[0], prefix);
          (k, v)
        }),
    )
  }

  fn multi_get(&self, keys: Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>> {
    let mut opts = ReadOptions::default();
    opts.set_async_io(true);
    self
      .db
      .multi_get_opt(keys, &opts)
      .into_iter()
      .map(|r| r.unwrap())
      .collect()
  }

  fn write(&self, ops: Vec<WriteOp>) {
    let mut batch = WriteBatch::default();
    for op in ops {
      match op {
        WriteOp::Put(key, value) => batch.put(key, value),
        WriteOp::Delete(key) => batch.delete(key),
      }
    }
    self.db.write(batch).unwrap();
  }
}
