from argparse import ArgumentParser
from tqdm import tqdm
from util import batches
from util import read_vecs
import cuml
import cupy as cp
import math
import msgpack
import numpy as np

parser = ArgumentParser()
parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
parser.add_argument("--out", type=str, help="Graph output path")
parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
parser.add_argument("--dim", type=int, help="Vector dimensions")
parser.add_argument("--m", type=int, help="Degree bound")
parser.add_argument("--k", type=int, help="Clusters")
parser.add_argument("--batch", type=int, help="Batch size", default=1000)
args = parser.parse_args()

print("Loading vectors")
dtype = np.dtype(args.dtype)
vecs = read_vecs(args.vectors, args.dim, dtype)
n, dim = vecs.shape
vecs = cp.asarray(vecs)
print("Copied to GPU:", vecs.device)

k = args.k
degree_bound = args.m
kmeans = cuml.KMeans(n_clusters=k)
cluster_labels = kmeans.fit_predict(vecs)
print("Clustered vectors:", cluster_labels.shape)

# Node ID => its neighbors.
graph_l0 = np.full((n, degree_bound), -1, dtype=np.int32)

for cluster_no in tqdm(range(k)):
    indices = cp.nonzero(cluster_labels == cluster_no)[0]
    cluster_size = indices.shape[0]
    cluster_vecs = vecs[indices]
    # Some clusters are larger than others, don't OOM by using batching.
    for start, end, batch_size in batches(cluster_size, args.batch):
        batch_indices = indices[start:end]
        batch_vecs = vecs[batch_indices]
        dists = cp.linalg.norm(batch_vecs[:, None] - cluster_vecs, axis=2)
        closest_indices = cp.argsort(dists, axis=1)[:, :degree_bound]
        # There may be fewer than degree_bound neighbors because there are fewer than degree_bound points in the cluster.
        graph_l0[cp.asnumpy(batch_indices), : closest_indices.shape[1]] = cp.asnumpy(
            closest_indices
        )
print("Built level 0")

# Since descending levels must have at least the above nodes, we sample by using a prefix of a specific fixed permutation of indices.
shuffled_indices = np.random.permutation(n).astype(np.uint32)
level_graphs = []
for i in range(1, math.floor(math.log(n, degree_bound)) + 1):
    level_size = math.floor(n * (degree_bound**-i) * (1 - 1 / degree_bound))
    graph = np.full((level_size, degree_bound), -1, dtype=np.int32)
    print(f"Level {i}: {level_size} nodes")
    indices = shuffled_indices[:level_size]
    level_vecs = vecs[indices]
    # Avoid OOM in larger levels.
    for start, end, batch_size in batches(level_size, 1000):
        batch_indices = indices[start:end]
        batch_vecs = vecs[batch_indices]
        dists = cp.linalg.norm(batch_vecs[:, None] - level_vecs, axis=2)
        closest_indices = cp.argsort(dists, axis=1)[:, :degree_bound]
        # There may be fewer than degree_bound neighbors because there are fewer than degree_bound points in the level.
        graph[start:end, : closest_indices.shape[1]] = cp.asnumpy(closest_indices)
    level_graphs.append(
        {
            "level": i,
            # Array of IDs of the nodes in this level.
            "nodes": indices.tobytes(),
            # Vector for each node in `nodes` containing the IDs of its neighbors.
            "neighbors_for_each_node": graph.tobytes(),
        }
    )
print("Built all levels")

with open(args.out, "wb") as f:
    msgpack.pack(
        {
            "l0_graph": graph_l0.tobytes(),
            "level_graphs": level_graphs,
            "entry_node": shuffled_indices[0].item(),
        },
        f,
    )

print("All done!")
