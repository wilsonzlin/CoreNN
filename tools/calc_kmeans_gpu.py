from collections import defaultdict
from tqdm import tqdm
import argparse
import cuml
import cupy as cp
import msgpack
import numpy as np

parser = argparse.ArgumentParser(
    description="[CUDA] Perform k-means clustering on vectors and calculate distance from each vector to each centroid."
)
parser.add_argument("--dim", type=int, help="Dimension of vectors")
parser.add_argument("--k", type=int, help="Clusters")
parser.add_argument(
    "--out-clusters",
    type=str,
    help="Path to Dict[ClusterId, List[VectorRow]] MessagePack output",
)
parser.add_argument(
    "--out-dists",
    type=str,
    help="Path to packed (n, k) matrix of f16 vector–centroid distances output",
)
parser.add_argument(
    "--vectors", type=str, help="Path to packed (n, dim) matrix of f32 vectors"
)
args = parser.parse_args()


def batches(n: int, sz: int):
    for start in range(0, n, sz):
        end = min(start + sz, n)
        # Last batch may be smaller than `sz`.
        yield (start, end, end - start)


with open(args.vectors, "rb") as f:
    vecs = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, args.dim))
n = vecs.shape[0]
print("Read vectors:", vecs.shape)
vecs = cp.asarray(vecs)
print("Copied vectors to GPU")

pb = tqdm(total=n)
k = args.k
kmeans_max_iter = 500
kmeans = cuml.KMeans(
    n_clusters=k,
    max_iter=kmeans_max_iter,
)

# Group vectors into clusters.
cluster_labels = kmeans.fit_predict(vecs)
cluster_members = defaultdict(list)
for id, label in enumerate(cluster_labels.tolist()):
    cluster_members[label].append(id)
with open(args.out_clusters, "wb") as f:
    f.write(msgpack.packb(cluster_members))

# Calculate distances to centroids.
with open(args.out_dists, "wb") as out_dists:
    for start, end, batch_size in batches(n, 1000):
        distances = cp.linalg.norm(
            vecs[start:end, None, :] - kmeans.cluster_centers_, axis=2
        )  # Shape: (batch_size, k)
        # Save space by converting to float16.
        distances = cp.asnumpy(distances).astype(np.float16)
        out_dists.write(distances.tobytes())
        pb.update(batch_size)
pb.close()
print("All done!")
