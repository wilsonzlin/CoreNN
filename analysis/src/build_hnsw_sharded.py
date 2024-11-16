from pathlib import Path
from tqdm import tqdm
import argparse
import hnswlib
import numpy as np
import os
import toml

parser = argparse.ArgumentParser(description="Build HNSW index shards.")
parser.add_argument("--m", type=int, help="M hyperparameter")
parser.add_argument("--ef", type=int, help="ef hyperparameter")
args = parser.parse_args()

dataset = os.environ["DS"]
out_dir = f"hnsw-sharded-{args.m}M-{args.ef}ef"

with open(f"dataset/{dataset}/info.toml", "r") as f:
    info = toml.load(f)
dim = info["dim"]

Path(f"dataset/{dataset}/out/{out_dir}/indices").mkdir(parents=True, exist_ok=True)

print("Loading")
mat = np.frombuffer(
    open(f"dataset/{dataset}/vectors.bin", "rb").read(), np.float32
).reshape((-1, dim))
n, _ = mat.shape

ids = np.arange(n)
np.random.seed(42)
np.random.shuffle(ids)
shards = 50
assert n % shards == 0
shard_size = n // shards

for i in tqdm(range(shards)):
    shard_ids = ids[i * shard_size : (i + 1) * shard_size]
    index = hnswlib.Index("l2", dim)
    index.init_index(max_elements=shard_size, ef_construction=args.ef, M=args.m)
    index.add_items(mat[shard_ids, :], shard_ids)
    index.save_index(f"dataset/{dataset}/out/{out_dir}/indices/shard{i}.hnsw")
print("All done!")
