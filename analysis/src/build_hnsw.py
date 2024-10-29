import hnswlib
import numpy as np
import pathlib

print("Loading")
with open("dataset/base.fvecs", "rb") as f:
    raw = f.read()
dim = int.from_bytes(raw[:4], byteorder="little")
n = len(raw) // (4 + dim * 4)
mat = np.vstack(
    [
        np.frombuffer(raw, dtype=np.float32, count=dim, offset=(4 + dim * 4) * i + 4)
        for i in range(n)
    ]
)

pathlib.Path("out/hnsw").mkdir(parents=True, exist_ok=True)

index = hnswlib.Index("l2", dim)
# Keep ef_construction the same as Vamana's search_list_cap and M as Vamana's degree bound.
index.init_index(max_elements=n, ef_construction=133, M=16)
print("Indexing")
index.add_items(mat, np.arange(n))
print("Saving")
index.save_index("out/hnsw/index.hnsw")
print("All done!")
