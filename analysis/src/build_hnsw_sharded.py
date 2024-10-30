from util import read_vectors
import hnswlib
import numpy as np
import pathlib

print("Loading")
mat = read_vectors("dataset/base.fvecs", np.float32)
n, dim = mat.shape

pathlib.Path("out/hnsw-sharded/indices").mkdir(parents=True, exist_ok=True)

ids = np.arange(n)
np.random.seed(42)
np.random.shuffle(ids)
shards = 50
assert n % shards == 0
shard_size = n // shards

for i in range(0, shards):
    shard_ids = ids[i * shard_size : (i + 1) * shard_size]
    index = hnswlib.Index("l2", dim)
    index.init_index(max_elements=shard_size, ef_construction=133, M=16)
    print("Indexing", i)
    index.add_items(mat[shard_ids, :], shard_ids)
    print("Saving", i)
    index.save_index(f"out/hnsw-sharded/indices/shard{i}.hnsw")
print("All done!")
