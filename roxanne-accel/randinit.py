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
parser.add_argument(
    "--batch", type=int, help="Optionally batch; slower but uses less memory"
)
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
batch = args.batch
dist_thresh = np.bfloat16(args.alpha)
seed = 0
rk = rand.PRNGKey(seed)
print(f"{n=} {m=} {m_max=} {ef=} {it=} {batch=} {dist_thresh=}")


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
    # TODO We also want to minimise getting stuck in local minima and not exploiting new/unseen/novel nodes i.e. keep picking the same nodes each iteration. (Although this is not 100% terrible as even seen nodes may have new neighbors between iterations.) Analyze this.
    # We pick the first `ef` ordered in anti-diagonal order as that approximates the closest nodes in `candidates`,
    # which we hope will converge faster to the true nearest neighbors. (Once again this is a heuristic, not a guarantee.)
    candidates = flatten_by_anti_diagonals(candidates)  # (m * (1 + m),)
    # Filter self-edges.
    candidates = np.where(candidates == node, NULL_ID, candidates)
    # Add a NULL_ID to ensure our later `np.argmax(candidates == NULL_ID)` doesn't return a false positive.
    # Unfortunately, the JAX implementation of np.unique(return_index=True) does not pad the returned indices with NULL_ID,
    # but instead with the index of the smallest unique value repeatedly.
    #           0  1  2     3  4     5  6  7  8     9  10
    # Example: [1, 3, 5, NULL, 5, NULL, 2, 4, 1, NULL, -1]
    candidates = np.hstack([candidates.astype(np.int32), NULL_ID, np.int32(-1)])
    # return_index to preserve ordering. We can't limit size=ef as that doesn't guarantee closest ef elements.
    # Example: [9, 0, 6, 1, 7, 2, 3, 9, 9, 9, 9]
    uniq_i = np.unique(
        candidates, return_index=True, size=candidates.shape[0], fill_value=NULL_ID
    )[1]
    # Index of the first NULL. We only need the first because np.unique() always returns the first index for it in `uniq_i`.
    # Since we previously inserted a NULL_ID, this will never be a false positive (i.e. 0 when there are no NULL_IDs)
    cand_null_i = np.argmax(candidates == NULL_ID)
    # Remove the index of the first NULL so it gets sorted to the end and unlikely to be picked.
    # Example: [9, 0, 6, 1, 7, 2, NULL, 9, 9, 9, 9]
    uniq_i = np.where(uniq_i != cand_null_i, uniq_i, NULL_ID)
    # Sort the indices so we get back our original optimal candidates order, now with duplicates filtered.
    # Example: [0, 1, 2, 6, 7, 9, 9, 9, 9, 9, NULL]
    uniq_i = np.sort(uniq_i)
    # Drop our dummy NULL and -1. Now, attempts to access them should return NULL_ID if used with `.get("fill", NULL_ID)`.
    # Example: [1, 3, 5, NULL, 5, NULL, 2, 4, 1]
    candidates = candidates[:-2].astype(np.uint32)
    # Retrieve our original candidates in the optimal order, without duplicates or self-edges.
    # Example: [1, 3, 5, 2, 4, NULL, NULL, NULL, NULL, NULL]
    candidates = candidates.at[uniq_i].get(mode="fill", fill_value=NULL_ID)
    # The code above may be a mind bender, but note: do not sort the candidates here, we've carefully preserved the order intentionally.
    candidates = candidates[:ef]  # (ef,)
    dists = norm(vecs[node] - select_vecs(vecs, candidates), axis=1)  # (ef,)
    sort_i = np.argsort(dists)
    return candidates[sort_i[:m]]


optimize_graph_batch = jax.vmap(
    optimize_graph_node, in_axes=(None, None, 0, None), out_axes=0
)


@partial(jax.jit, static_argnames=("n", "ef", "batch"))
def optimize_graph_batched(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    n: int,
    ef: int,
    batch: int,
):
    """
    Updating in batches is slower than all at once. Avoid this function when possible.
    """
    # Updating batches then writing back after each batch is even worse, as that requires a memory barrier/sync between batches.
    nodes = arange(n)
    assert n % batch == 0
    num_batches = n % batch
    def loop_body(i, new_graph):
        start = i * batch
        batch_nodes = jax.lax.dynamic_slice(nodes, (start,), (batch,))
        batch_res = optimize_graph_batch(graph, vecs, batch_nodes, ef)
        return jax.lax.dynamic_update_slice(new_graph, batch_res, (start,))
    new_graph = np.zeros(graph.shape, dtype=graph.dtype)
    # Don't use Python loop as that will get unrolled and not reduce memory usage.
    return jax.lax.fori_loop(0, num_batches, loop_body, new_graph)


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
    vecs: BFloat16[Array, "n d"],
    n: int,
    m: int,
    seed: int,
):
    rk = rand.PRNGKey(seed)
    graph = rand.choice(rk, arange(n), shape=(n, m), replace=True)  # (n, m)
    # (n, 1, d) - (n, m, d) = (n, m, d)
    dists = norm(vecs[:, None] - select_vecs(vecs, graph), axis=2)  # (n, m)
    sort_i = np.argsort(dists, axis=1)  # (n, m)
    graph = graph[arange(n)[:, None], sort_i]  # (n, m)
    return graph


print("Initializing graph")
graph = init_graph(vecs=vecs, n=n, m=m_max, seed=seed)
print("Compiling optimizer")
if batch is None:
    opt_fn = optimize_graph.lower(graph=graph, vecs=vecs, n=n, ef=ef).compile()
else:
    opt_fn = optimize_graph_batched.lower(
        graph=graph, vecs=vecs, n=n, ef=ef, batch=batch
    ).compile()
print(
    "Optimization cost per iteration:",
    opt_fn.cost_analysis()[0]["flops"] / 1e12,
    "TFLOPS",
)
for i in tqdm(range(it), desc="Optimizing graph"):
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
