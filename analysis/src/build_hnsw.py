from util import read_vectors
import hnswlib
import numpy as np
import pathlib

print("Loading")
mat = read_vectors("dataset/base.fvecs", np.float32)
n, dim = mat.shape

pathlib.Path("out/hnsw").mkdir(parents=True, exist_ok=True)

index = hnswlib.Index("l2", dim)
# Keep ef_construction the same as Vamana's search_list_cap and M as Vamana's degree bound.
index.init_index(max_elements=n, ef_construction=133, M=16)
print("Indexing")
index.add_items(mat, np.arange(n))
print("Saving")
index.save_index("out/hnsw/index.hnsw")
print("All done!")
