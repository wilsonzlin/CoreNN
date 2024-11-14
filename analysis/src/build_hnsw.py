import hnswlib
import numpy as np
import os

dataset = os.environ["DS"]
out_dir = os.environ["OUT"]
dim = int(os.environ["DIM"])
m = int(os.environ["M"])
ef = int(os.environ["EF"])

print("[build_hnsw] Loading")
mat = np.frombuffer(
    open(f"dataset/{dataset}/vectors.bin", "rb").read(), np.float32
).reshape((-1, dim))
n, _ = mat.shape

index = hnswlib.Index("l2", dim)
# Keep ef_construction the same as Vamana's search_list_cap and M as Vamana's degree bound.
index.init_index(max_elements=n, ef_construction=ef, M=m)
print("[build_hnsw] Indexing")
index.add_items(mat, np.arange(n))
print("[build_hnsw] Saving")
index.save_index(f"dataset/{dataset}/out/{out_dir}/index.hnsw")
print("[build_hnsw] All done!")
