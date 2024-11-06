from util import read_vectors
import hnswlib
import numpy as np
import os

dataset = os.environ["DS"]

print("Loading queries")
mat = read_vectors(f"dataset/{dataset}/query.fvecs", np.float32)
n, dim = mat.shape
ground_truth = read_vectors(f"dataset/{dataset}/groundtruth.ivecs", np.uint32)
_, k = ground_truth.shape

index = hnswlib.Index("l2", dim)
print("Loading index")
index.load_index(f"dataset/{dataset}/out/hnsw/index.hnsw")
index.set_ef(index.ef_construction)
print("Querying")
knn, _ = index.knn_query(mat, k)

correct = 0
total = n * k
for i in range(n):
  correct += len(set(knn[i]).intersection(set(ground_truth[i])))
accuracy = (correct / total) * 100
print(f"Correct: {accuracy:.2f}% ({correct}/{total})")
