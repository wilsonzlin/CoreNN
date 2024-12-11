from argparse import ArgumentParser
from functools import partial
from index import calc_approx_medoid
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import BFloat16
from jaxtyping import UInt32
from robust_prune import compute_robust_pruned
from tqdm import tqdm
from util import arange
from util import batches
from util import NULL_ID
from util import read_vecs
import jax
import jax.numpy as np
import jax.random as rand

parser = ArgumentParser(description="Build a RandInit graph using the GPU.")
parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
parser.add_argument("--out", type=str, help="Graph output path")
parser.add_argument("--out-medoid", type=str, help="Medoid output path")
parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
parser.add_argument("--dim", type=int, help="Vector dimensions")
parser.add_argument("--m", type=int, help="Degree bound")
parser.add_argument("--ef", type=int, help="Search list cap")
parser.add_argument("--iter", type=int, help="Update iterations")
parser.add_argument("--alpha", type=float, help="Distance threshold", default=1.1)
parser.add_argument("--batch", type=int, help="Batch size", default=1000)
args = parser.parse_args()

print("Loading vectors")
dtype = np.dtype(args.dtype)
vecs = read_vecs(args.vectors, args.dim, dtype)
n, _ = vecs.shape
assert n < NULL_ID
m = args.m
ef = args.ef
it = args.iter
batch = args.batch
dist_thresh = np.bfloat16(args.alpha)
seed = 0
rk = rand.PRNGKey(seed)
medoid_sample_size = 10_000
print(f"{n=} {m=} {ef=} {it=} {batch=} {dist_thresh=}")

print("Calculating approx. medoid")
medoid = calc_approx_medoid(
    vecs=vecs,
    sample_size=medoid_sample_size,
    seed=seed,
)


@jax.jit
def optimize_graph_node(
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    node: UInt32[Array, ""],
):
    _, m = graph.shape
    neighbors = graph[node]  # (m,)
    neighbor_neighbors = graph[neighbors].flatten()  # (m * m,)
    candidates = np.concatenate([neighbors, neighbor_neighbors])  # (c = m + m * m,)
    candidates = np.unique(candidates, size=candidates.size, fill_value=NULL_ID)  # (c,)
    dists = norm(vecs[node] - vecs[candidates], axis=1)  # (c,)
    sort_i = np.argsort(dists)
    return candidates[sort_i][:m]


optimize_graph_batch = jax.jit(
    jax.vmap(optimize_graph_node, in_axes=(None, None, 0), out_axes=0)
)


@partial(jax.jit, static_argnames=("n", "batch"))
def optimize_graph(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    n: int,
    batch: int,
):
    nodes = arange(n)
    for start, end, b in batches(n, batch):
        graph = graph.at[start:end].set(
            optimize_graph_batch(graph, vecs, nodes[start:end])
        )
    return graph


@partial(jax.jit, static_argnames=("n", "ef", "seed"))
def init_graph(
    *,
    n: int,
    ef: int,
    seed: int,
):
    rk = rand.PRNGKey(seed)
    graph = rand.choice(rk, arange(n), shape=(n, ef), replace=True)
    return graph


print("Initializing graph")
graph = init_graph(n=n, ef=ef, seed=seed)
print("Compiling optimizer")
optimize_graph.lower(graph=graph, vecs=vecs, n=n, batch=batch).compile()
for i in tqdm(range(it), desc="Optimizing graph"):
    # Batching (instead of just doing all nodes at once) not only helps with larger datasets and/or smaller GPUs,
    # but also increases chance of finding better neighbors within an iteration, as later batches may use previous intra-batch updates.
    # block_until_ready() for more accurate progress.
    graph = optimize_graph(graph=graph, vecs=vecs, n=n, batch=batch).block_until_ready()
print("Final prune")
graph = compute_robust_pruned(
    vecs=vecs,
    node_ids=arange(n),
    cand_ids=graph,
    m=m,
    dist_thresh=dist_thresh,
).block_until_ready()
print("Saving", graph.dtype, graph.shape)
with open(args.out, "wb") as f:
    # WARNING: Do not convert to Python type and serialize as MessagePack/JSON/etc. as that conversion + serialization process will be extremely slow in Python.
    f.write(graph.tobytes())
with open(args.out_medoid, "w") as f:
    f.write(str(medoid.item()))
print("All done!")
