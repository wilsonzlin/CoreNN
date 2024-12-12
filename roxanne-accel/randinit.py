from argparse import ArgumentParser
from functools import partial
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import BFloat16
from jaxtyping import UInt32
from robust_prune import compute_robust_pruned
from tqdm import tqdm
from util import arange
from util import batches
from util import flatten_by_anti_diagonals
from util import NULL_ID
from util import read_vecs
from util import select_nodes
from util import select_vecs
import jax
import jax.numpy as np
import jax.random as rand
import math
import msgpack

parser = ArgumentParser(description="Build a RandInit graph using the GPU.")
parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
parser.add_argument("--out", type=str, help="Graph output path")
parser.add_argument("--out-levels", type=str, help="Level graphs output path")
parser.add_argument("--out-medoid", type=str, help="Medoid output path")
parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
parser.add_argument("--dim", type=int, help="Vector dimensions")
parser.add_argument("--m", type=int, help="Degree bound")
parser.add_argument("--m-max", type=int, help="Maximum degree bound")
parser.add_argument("--ef", type=int, help="Search list cap")
parser.add_argument("--iter", type=int, help="Update iterations")
parser.add_argument("--alpha", type=float, help="Distance threshold", default=1.1)
args = parser.parse_args()

print("Loading vectors")
dtype = np.dtype(args.dtype)
vecs = read_vecs(args.vectors, args.dim, dtype)
n, _ = vecs.shape
assert n < NULL_ID
m = args.m
m_max = args.m_max
ef = args.ef
it = args.iter
dist_thresh = np.bfloat16(args.alpha)
seed = 0
rk = rand.PRNGKey(seed)
print(f"{n=} {m=} {m_max=} {ef=} {it=} {dist_thresh=}")


def optimize_graph_node(
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    node: UInt32[Array, ""],
    ef: int,
):
    _, m = graph.shape
    neighbors = graph[node]  # (m,)
    # Neighbors can be NULL_ID due to np.unique after a few iterations.
    neighbor_neighbors = select_nodes(graph, neighbors)  # (m, m)
    candidates = np.vstack([neighbors[None], neighbor_neighbors])  # (1 + m, m)
    # NOTE: Esp. in high dim. spaces, it's possible for a far neighbor N to have its neighbor U actually be closer to us than N,
    # so this is merely a heuristic, that we hope is better than these other strategies:
    # - Randomly pick X neighbor-neighbors of **every** neighbor (even far ones).
    # - Pick the first (i.e. closest) X neighbor-neighbors of **every** neighbor (even far ones).
    # - Pick all neighbor-neighbors of the first (i.e. closest) X neighbors.
    # Basically, strategies have two axes:
    # 1) Which neighbors to pick? All, first/closest X, random X.
    # 2) Of those neighbors, which of their neighbors (i.e. neighbor-neighbors) to pick? All, first/closest X, random X.
    # Alternatively, it's possible to pick all/first-X/random-X across all neighbors and neighbor-neighbors as a flat list, which is what we're doing (first X).
    # TODO We also want to minimise getting stuck in local minima and not exploiting new/unseen/novel nodes i.e. keep picking the same nodes each iteration. Analyze this.
    # We pick the first `ef` ordered in anti-diagonal order as that approximates the closest nodes in `candidates`,
    # which we hope will converge faster to the true nearest neighbors. (Once again this is a heuristic, not a guarantee.)
    candidates = flatten_by_anti_diagonals(candidates)  # (m * (1 + m),)
    # Filter self-edges.
    candidates = np.where(candidates == node, NULL_ID, candidates)
    # return_index to preserve ordering. We can't limit size=ef as that doesn't guarantee closest ef elements.
    uniq_i = np.unique(
        candidates, return_index=True, size=candidates.shape[0], fill_value=NULL_ID
    )[1]
    candidates = candidates.at[np.sort(uniq_i)].get(mode="fill", fill_value=NULL_ID)[
        :ef
    ]  # (ef,)
    dists = norm(vecs[node] - select_vecs(vecs, candidates), axis=1)  # (ef,)
    sort_i = np.argsort(dists)
    return candidates[sort_i[:m]]


optimize_graph_batch = jax.vmap(
    optimize_graph_node, in_axes=(None, None, 0, None), out_axes=0
)


@partial(jax.jit, static_argnames=("n", "ef"))
def optimize_graph(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    n: int,
    ef: int,
):
    # Updating in batches is slower than all at once.
    # Updating batches then writing back after each batch is even worse, as that requires a memory barrier/sync between batches.
    return optimize_graph_batch(graph, vecs, arange(n), ef)


@partial(jax.jit, static_argnames=("n", "m", "seed"))
def init_graph(
    *,
    n: int,
    m: int,
    seed: int,
):
    rk = rand.PRNGKey(seed)
    graph = rand.choice(rk, arange(n), shape=(n, m), replace=True)
    return graph


print("Initializing graph")
graph = init_graph(n=n, m=m_max, seed=seed)
print("Compiling optimizer")
opt_fn = optimize_graph.lower(graph=graph, vecs=vecs, n=n, ef=ef).compile()
print(
    "Optimization cost per iteration:",
    opt_fn.cost_analysis()[0]["flops"] / 1e12,
    "TFLOPS",
)
for i in tqdm(range(it), desc="Optimizing graph"):
    # Batching (instead of just doing all nodes at once) helps with larger datasets and/or smaller GPUs.
    # block_until_ready() for more accurate progress.
    graph = opt_fn(graph=graph, vecs=vecs).block_until_ready()


# Since descending levels must have at least the above nodes, we sample by using a prefix of a specific fixed permutation of indices.
shuffled_nodes = rand.permutation(rk, arange(n))
level_graphs = []
for i in range(1, math.floor(math.log(n, m)) + 1):
    level_size = math.floor(n * (m**-i) * (1 - 1 / m))
    print(f"Level {i}: {level_size} nodes")
    level_m = min(m, level_size)
    level_g = np.full((level_size, level_m), NULL_ID, dtype=np.uint32)
    level_nodes = shuffled_nodes[:level_size]
    level_vecs = vecs[level_nodes]
    # Avoid OOM in larger levels.
    for start, end, batch_size in batches(level_size, 1000):
        batch_nodes = level_nodes[start:end]
        batch_vecs = vecs[batch_nodes]
        dists = norm(batch_vecs[:, None] - level_vecs, axis=2)
        sort_i = np.argsort(dists, axis=1)[:, :level_m]
        level_g = level_g.at[start:end].set(level_nodes[sort_i])
    level_graphs.append(
        {
            "level": i,
            # Array of IDs of the nodes in this level.
            "nodes": level_nodes.tobytes(),
            # Vector for each node in `nodes` containing the IDs of its neighbors.
            "neighbors_for_each_node": level_g.tobytes(),
        }
    )
entry_node = shuffled_nodes[0]


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
with open(args.out_levels, "wb") as f:
    f.write(msgpack.packb(level_graphs))
with open(args.out_medoid, "w") as f:
    f.write(str(entry_node.item()))
print("All done!")
