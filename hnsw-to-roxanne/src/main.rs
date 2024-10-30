use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use hnswlib_rs::LabelType;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne::vamana::VamanaParams;
use lmdb::DatabaseFlags;
use lmdb::Environment;
use lmdb::Transaction;
use lmdb::WriteFlags;
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a text file containing paths to HNSW shards.
  #[arg()]
  paths: PathBuf,

  /// Output directory to write Roxanne index to.
  #[arg(long)]
  out: PathBuf,

  /// How many shards (n) to compute in parallel; requires n shards in memory at one time and n threads.
  #[arg(long, default_value_t = 2)]
  parallel: usize,

  /// Dimensions of the vectors.
  #[arg(long)]
  dim: usize,
}

fn load_hnsw(dim: usize, path: impl AsRef<Path>) -> HnswIndex {
  let raw = File::open(path).unwrap();
  let mut rd = BufReader::new(raw);
  HnswIndex::load(dim, &mut rd)
}

pub struct HnswDatastore {
  hnsw: HnswIndex,
}

impl HnswDatastore {
  pub fn new(hnsw: HnswIndex) -> Self {
    Self { hnsw }
  }
}

impl VamanaDatastore<f32> for HnswDatastore {
  fn get_point(&self, id: Id) -> Option<Array1<f32>> {
    let vec = self.hnsw.get_data_by_label(id);
    Some(Array1::from_vec(vec))
  }

  fn set_point(&self, _id: Id, _point: Array1<f32>) {
    panic!("read only");
  }

  fn get_out_neighbors(&self, id: Id) -> Option<HashSet<Id>> {
    Some(self.hnsw.get_merged_neighbors(id, 0))
  }

  fn set_out_neighbors(&self, _id: Id, _neighbors: HashSet<Id>) {
    panic!("read only");
  }
}

fn into_vamana(hnsw: HnswIndex) -> Vamana<f32, HnswDatastore> {
  let medoid = hnsw.entry_label();
  let params = VamanaParams {
    beam_width: 1,
    degree_bound: hnsw.m,
    distance_threshold: 1.1,
    update_batch_size: num_cpus::get(),
    update_search_list_cap: hnsw.ef_construction,
  };
  let ds = HnswDatastore::new(hnsw);
  Vamana::new(ds, metric_euclidean, medoid, params)
}

fn new_pb(len: usize) -> ProgressBar {
  let pb = ProgressBar::new(len.try_into().unwrap());
  pb.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-"),
  );
  pb
}

// We don't perform PQ here, as it's expensive and not specifically related to this task; use the general `pqify` tool afterwards.
fn main() {
  let args = Args::parse();

  // Make sure database can be created before we do long expensive work.
  let db_env = Environment::new().open(&args.out).unwrap();
  let db = db_env.create_db(None, DatabaseFlags::empty()).unwrap();
  println!("Created database");

  // First phase: load each shard to ensure integrity and compute some things.
  // - Ensure all shard files exist and are intact before we start doing long intensive work.
  // - Ensure all IDs are unique across all shards.
  // - Sample subset of embeddings across all shards to build PQ.
  let paths_raw = std::fs::read_to_string(&args.paths).unwrap();
  let mut paths = paths_raw.lines().collect::<Vec<_>>();
  let shard_sizes = DashMap::<&str, usize>::new();
  let seen_ids = DashMap::<Id, String>::new();
  let pb = new_pb(paths.len());
  paths.par_iter().for_each(|path| {
    let index = load_hnsw(args.dim, path);
    shard_sizes.insert(path, index.cur_element_count);
    for id in index.labels() {
      if let Some(other) = seen_ids.insert(id, path.to_string()) {
        panic!("Duplicate ID {id} in {other} and {path}");
      };
    }
    pb.inc(1);
  });
  let total_n = shard_sizes.iter().map(|e| *e.value()).sum::<usize>();
  pb.finish_with_message("Verified shards");

  // Map from base node ID to combined out-neighbors.
  // We don't need a set, as a node cannot have an out-neighbor that also exists in another shard.
  let cliques = DashMap::<Id, Vec<Id>>::new();
  let node_to_clique = DashMap::<Id, Id>::new();
  // We'll preserve the amount of out-neighbors each node has in case that's important.
  let out_neighbor_counts = DashMap::<Id, usize>::new();

  // Use the largest shard as the base, as otherwise there may be some leftovers and the code/algorithm gets messy. (Shards may not be perfectly balanced.)
  paths.sort_unstable_by_key(|p| *shard_sizes.get(p).unwrap());
  let base = load_hnsw(args.dim, &paths[0]);
  let cfg = VamanaParams {
    beam_width: 4,
    degree_bound: base.m,
    distance_threshold: 1.1,
    update_batch_size: 64,
    update_search_list_cap: base.ef_construction,
  };
  let medoid = base.enter_point_node;
  let mut base_ids_by_level = HashMap::<usize, Vec<LabelType>>::new();
  for id in base.labels() {
    cliques.insert(id, base.get_merged_neighbors(id, 0).into_iter().collect());
    node_to_clique.insert(id, id);
    let level = base.get_node_level(id);
    assert!(level <= base.max_level);
    base_ids_by_level.entry(level).or_default().push(id);
  }
  println!("Loaded base shard {}", &paths[0]);
  let pb = new_pb(total_n - base.cur_element_count);
  for path_batch in (&paths[1..]).chunks(args.parallel - 1) {
    path_batch.into_par_iter().for_each(|path| {
      let mut second = into_vamana(load_hnsw(args.dim, path));
      second.params_mut().update_search_list_cap = 1;
      // This should persist across levels.
      let mut seen = HashSet::new();
      for level in (0..=base.max_level).rev() {
        // We only want to pick fellow nodes on the same level in other shards for our clique.
        second.datastore().set_level(level);
        for &id in base_ids_by_level.get(&level).unwrap().iter() {
          let emb = Array1::from_vec(base.get_data_by_label(id));
          let Some(nn) = second
            .query_with_filter(&emb.view(), 1, |e| !seen.contains(&e.id))
            .pop()
          else {
            continue;
          };
          pb.inc(1);
          seen.insert(nn.id);
          // Get full out-neighbors, not filtered by HnswDatastore.min_level.
          let neighbors = second.datastore().hnsw.get_merged_neighbors(nn.id, 0);
          out_neighbor_counts.insert(nn.id, neighbors.len());
          cliques.get_mut(&id).unwrap().extend(neighbors.iter());
          node_to_clique.insert(nn.id, id);
        }
      }
    });
  }
  drop(base);
  pb.finish_with_message("Processed shards");

  // Shuffle clique out-neighbors.
  cliques.par_iter_mut().for_each(|mut e| {
    e.shuffle(&mut thread_rng());
  });
  println!("Shuffled edges");

  {
    let mut txn = db_env.begin_rw_txn().unwrap();
    txn
      .put(
        db,
        &"cfg".to_string(),
        &rmp_serde::to_vec_named(&cfg).unwrap(),
        WriteFlags::empty(),
      )
      .unwrap();
    txn
      .put(
        db,
        &"dim".to_string(),
        &args.dim.to_le_bytes(),
        WriteFlags::empty(),
      )
      .unwrap();
    txn
      .put(
        db,
        &"medoid".to_string(),
        &medoid.to_le_bytes(),
        WriteFlags::empty(),
      )
      .unwrap();
  }

  let pb = new_pb(total_n);
  for path_batch in paths.chunks(args.parallel) {
    path_batch.into_par_iter().for_each(|path| {
      let index = load_hnsw(args.dim, path);
      index.labels().par_bridge().for_each(|id| {
        let clique_id = node_to_clique.get(&id).unwrap();
        // Release lock ASAP.
        let new_out_neighbors = {
          let mut clique = cliques.get_mut(&clique_id).unwrap();
          let n = clique.len();
          // We've already shuffled this Vec, so we can just take the last n cheaply.
          let pos = n
            .checked_sub(*out_neighbor_counts.get(&id).unwrap())
            .unwrap();
          clique.split_off(pos)
        };

        let mut node_data = Vec::new();
        node_data.extend(new_out_neighbors.len().to_le_bytes());
        for &n in new_out_neighbors.iter() {
          node_data.extend(n.to_le_bytes());
        }
        node_data.extend(index.get_raw_data_by_internal_id(index.get_internal_id(id)));

        let mut txn = db_env.begin_rw_txn().unwrap();
        txn
          .put(db, &format!("node/{id}"), &node_data, WriteFlags::empty())
          .unwrap();
        txn.commit().unwrap();
        pb.inc(1);
      });
    });
  }
  pb.finish_with_message("All done!");
}
