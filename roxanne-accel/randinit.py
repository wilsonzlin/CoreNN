from argparse import ArgumentParser
from functools import partial
from greedy_search import greedy_search
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import BFloat16
from jaxtyping import UInt32
from robust_prune import compute_robust_pruned
from tqdm import tqdm
from util import arange
from util import batches
from util import group_by_y_id_limit
from util import NULL_ID
from util import read_vecs
from util import select_nodes
from util import select_vecs
import jax
import jax.numpy as np
import jax.random as rand
import math
import msgpack


def optimize_l0_graph_node(
    graph: UInt32[Array, "n m"],
    backedges: UInt32[Array, "n r"],
    vecs: BFloat16[Array, "n d"],
    node: UInt32[Array, ""],
    ef: int,
    strat: str,
    rk,
):
    _, m = graph.shape
    neighbors = graph[node]  # (m,)
    # Neighbors can be NULL_ID due to np.unique after a few iterations.
    neighbor_neighbors = select_nodes(graph, neighbors)  # (m, m)
    # NOTE: Esp. in high dim. spaces, it's possible for a far neighbor N to have its neighbor U actually be closer to us than N,
    # so the strategy for picking ef is merely a heuristic. Some strategies include:
    # - Randomly pick X neighbor-neighbors of **every** neighbor (even far ones).
    # - Pick the first (i.e. closest) X neighbor-neighbors of **every** neighbor (even far ones).
    # - Pick all neighbor-neighbors of the first (i.e. closest) X neighbors.
    # Basically, strategies have two axes:
    # 1) Which neighbors to pick? All, first/closest X, random X.
    # 2) Of those neighbors, which of their neighbors (i.e. neighbor-neighbors) to pick? All, first/closest X, random X.
    neighbor_neighbors = neighbor_neighbors.flatten()  # (m * (1 + m),)
    # This should only affect neighbor-neighbors; our current neighbors and backedges should always be used.
    if strat == "F":
        pass
    elif strat == "L":
        neighbor_neighbors = np.flip(neighbor_neighbors)
    elif strat == "R":
        neighbor_neighbors = rand.permutation(rk, neighbor_neighbors)
    else:
        assert False
    # Add backedges as candidates (and prioritise them by adding to the front).
    candidates = np.hstack(
        [neighbors, backedges[node], neighbor_neighbors]
    )  # (m + r + m*m,)
    # Filter self-edges.
    candidates = np.where(candidates == node, NULL_ID, candidates)
    # Add a NULL_ID to ensure our later `np.argmax(candidates == NULL_ID)` doesn't return a false positive.
    # Unfortunately, the JAX implementation of np.unique(return_index=True) does not pad the returned indices with NULL_ID,
    # but instead with the index of the smallest unique value repeatedly; therefore, we add a -1 to distinguish padding
    # from the index of the true lowest value, as -1 will always be the lowest.
    #           0  1  2     3  4     5  6  7  8     9  10
    # Example: [1, 3, 5, NULL, 5, NULL, 2, 4, 1, NULL, -1]
    #                    ^        ^              ⬑ Inserted NULL, not existing one.
    #           Existing NULLs can come from the graph, or filtering out self-edges.
    candidates = np.hstack([candidates, NULL_ID, -1], dtype=np.int32)
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
    # All the above is so that we pick high quality `ef` candidates and not let any `ef` slot go to waste.
    candidates = candidates[:ef]  # (ef,)
    dists = norm(vecs[node] - select_vecs(vecs, candidates), axis=1)  # (ef,)
    sort_i = np.argsort(dists)
    return candidates[sort_i[:m]]


optimize_l0_graph_batch = jax.vmap(
    optimize_l0_graph_node, in_axes=(None, None, None, 0, None, None, None), out_axes=0
)


@partial(jax.jit, static_argnames=["r"])
def calc_backedges(
    graph: UInt32[Array, "n m"],
    r: int,
):
    n, m = graph.shape
    back_nodes, backedges = group_by_y_id_limit(
        M_id=arange(n),
        M=graph,
        d=r,
        u=n + 1,  # NULL_ID is a possible value.
    )
    sort_i = np.argsort(back_nodes)
    return backedges[sort_i]


@partial(jax.jit, static_argnames=["ef", "r", "batch", "strat"])
def optimize_l0_graph_batched(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    ef: int,
    r: int,
    batch: int,
    strat: str,
    rk,
):
    n, _ = graph.shape
    nodes = arange(n)
    assert n % batch == 0
    num_batches = n // batch

    # Compute at the start of each iteration, not end:
    # - Computing at end means we waste it on the last iteration, and don't use it in the first iteration (the graph is initialized sorted, so calc. backedges is useful even on the first iteration).
    # - We don't have to return it and receive it as a parameter.
    if r:
        backedges = calc_backedges(graph, r)
    else:
        print("Skipping calculation of backedges")
        backedges = np.full((n, 0), NULL_ID, dtype=np.uint32)

    # Updating batches then writing back after each batch is even worse, as that requires a memory barrier/sync between batches.
    def loop_body(i, new_graph):
        start = i * batch
        batch_nodes = jax.lax.dynamic_slice(nodes, (start,), (batch,))
        batch_res = optimize_l0_graph_batch(
            graph, backedges, vecs, batch_nodes, ef, strat, rk
        )
        return jax.lax.dynamic_update_slice(new_graph, batch_res, (start, 0))

    new_graph = np.zeros_like(graph)
    # Don't use Python loop as that will get unrolled and not reduce memory usage.
    return jax.lax.fori_loop(0, num_batches, loop_body, new_graph, unroll=False)


@partial(jax.jit, static_argnames=["n", "m"])
def init_l0_graph(
    *,
    vecs: BFloat16[Array, "n d"],
    n: int,
    m: int,
    rk,
):
    graph = rand.choice(rk, arange(n), shape=(n, m), replace=True)  # (n, m)
    # (n, 1, d) - (n, m, d) = (n, m, d)
    dists = norm(vecs[:, None] - select_vecs(vecs, graph), axis=2)  # (n, m)
    # If we don't sort now, our first iteration will pick the first ef neighbor-neighbors, assuming they're closest when they're not, and essentially pick more random nodes when it's already random, a waste of an iteration.
    sort_i = np.argsort(dists, axis=1)  # (n, m)
    graph = graph[arange(n)[:, None], sort_i]  # (n, m)
    return graph


@partial(jax.jit, static_argnames=["level_m"])
def build_level_graph(
    *,
    vecs: BFloat16[Array, "n d"],
    level_m: int,
    level_nodes: UInt32[Array, "n"],
    level_vecs: BFloat16[Array, "n d"],
):
    (level_size,) = level_nodes.shape
    level_g = np.full((level_size, level_m), NULL_ID, dtype=np.uint32)
    # Avoid OOM in larger levels.
    # TODO Use batch param.
    for start, end, _ in batches(level_size, 1000):
        batch_nodes = level_nodes[start:end]
        batch_vecs = vecs[batch_nodes]
        dists = norm(batch_vecs[:, None] - level_vecs, axis=2)
        sort_i = np.argsort(dists, axis=1)[:, :level_m]
        level_g = level_g.at[start:end].set(level_nodes[sort_i])
    return level_g


@partial(jax.jit, static_argnames=["m", "batch"])
def compute_robust_prune_batched(
    *,
    graph: UInt32[Array, "n m_max"],
    vecs: BFloat16[Array, "n d"],
    m: int,
    dist_thresh: BFloat16[Array, ""],
    batch: int,
):
    n, m_max = graph.shape
    nodes = arange(n)
    assert n % batch == 0
    num_batches = n // batch

    def loop_body(i, new_graph):
        start = i * batch
        batch_nodes = jax.lax.dynamic_slice(nodes, (start,), (batch,))
        batch_cand_ids = jax.lax.dynamic_slice(graph, (start, 0), (batch, m_max))
        batch_res = compute_robust_pruned(
            vecs=vecs,
            node_ids=batch_nodes,
            cand_ids=batch_cand_ids,
            m=m,
            dist_thresh=dist_thresh,
        )
        return jax.lax.dynamic_update_slice(new_graph, batch_res, (start, 0))

    new_graph = np.zeros(shape=(n, m), dtype=np.uint32)
    # Don't use Python loop as that will get unrolled and not reduce memory usage.
    return jax.lax.fori_loop(0, num_batches, loop_body, new_graph, unroll=False)


@partial(jax.jit, static_argnames=["k"])
def brute_force_knn(
    *,
    nodes: UInt32[Array, "s"],
    vecs: BFloat16[Array, "n d"],
    k: int,
):
    # (s, 1, d) - (1, n, d) = (s, n, d)
    dists = norm(vecs[nodes, None, :] - vecs[None], axis=2)  # (s, n)
    return np.argpartition(dists, k, axis=1)[:, :k]


def intersect_count(a: UInt32[Array, "a"], b: UInt32[Array, "b"]):
    return np.count_nonzero(np.isin(a, b))


intersect_count_batch = jax.jit(jax.vmap(intersect_count, in_axes=(0, 0), out_axes=0))


def main():
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
    parser.add_argument("--it", type=str, help="Strategy per iter (e.g. FFLRFLRF)")
    parser.add_argument("--r", type=int, help="Backedges to insert")
    parser.add_argument(
        "--eval-q", type=str, help="[Optional] Path to f32 query vectors"
    )
    parser.add_argument(
        "--eval-r", type=str, help="[Optional] Path to u32 query result vectors"
    )
    parser.add_argument(
        "--nn-samp", type=int, help="[Optional] Track true NNs of subset of nodes"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="[Optional] Enable batching of size; slower but uses less memory and doesn't affect results",
    )
    parser.add_argument("--alpha", type=float, help="Distance threshold", default=1.1)
    args = parser.parse_args()

    print("Loading vectors")
    dtype = np.dtype(args.dtype)
    dim = args.dim
    vecs = read_vecs(args.vectors, dim, dtype)
    n, _ = vecs.shape
    m = args.m
    m_max = args.m_max
    ef = args.ef
    r = args.r
    assert 1 <= m <= m_max <= ef <= m_max * m_max <= n < NULL_ID
    it = args.it
    batch = args.batch
    assert batch is None or n % batch == 0
    alpha = args.alpha
    dist_thresh = np.bfloat16(alpha)
    seed = 0
    rk = rand.key(seed)
    print(f"{n=} {m=} {m_max=} {ef=} {it=} {batch=} {alpha=}")

    # Since descending levels must have at least the above nodes, we sample by using a prefix of a specific fixed permutation of indices.
    shuffled_nodes = rand.permutation(rk, arange(n))
    # level_graphs[level - 1] = { "nodes": Array, "neighbors_for_each_node": Array }
    level_graphs = []
    for i in range(1, math.floor(math.log(n, m)) + 1):
        # We don't need to multiply by (1 - 1/m) as the level_size includes higher levels.
        level_size = math.floor(n * (m**-i))
        print(f"Level {i}: {level_size} nodes")
        level_m = min(m, level_size)
        level_nodes = shuffled_nodes[:level_size]
        level_vecs = vecs[level_nodes]
        level_g = build_level_graph(
            vecs=vecs,
            level_m=level_m,
            level_nodes=level_nodes,
            level_vecs=level_vecs,
        )
        level_graphs.append(
            {
                # Array of IDs of the nodes in this level.
                "nodes": level_nodes,
                # Vector for each node in `nodes` containing the IDs of its neighbors.
                "neighbors_for_each_node": level_g,
            }
        )
    entry_node = shuffled_nodes[0]

    print("Initializing L0 graph")
    l0_graph = init_l0_graph(vecs=vecs, n=n, m=m_max, rk=rk)
    l0_rks = rand.split(rk, len(it))
    pb = None
    if not (args.nn_samp or args.eval_q):
        pb = tqdm(total=len(it), desc="Optimizing L0 graph", unit="iterations")
    if args.nn_samp:
        nn_samp_nodes = shuffled_nodes[: args.nn_samp]
        nn_samp_knn = brute_force_knn(nodes=nn_samp_nodes, vecs=vecs, k=m_max)
        print(f"Calculated true k-NN for {args.nn_samp} vectors")
    if args.eval_q:
        with open(args.eval_q, "rb") as f:
            eval_queries = (
                np.frombuffer(f.read(), dtype=np.float32)
                .reshape((-1, dim))
                .astype(np.bfloat16)
            )
        eval_q_n, _ = eval_queries.shape
        with open(args.eval_r, "rb") as f:
            eval_results = np.frombuffer(f.read(), dtype=np.uint32).reshape(
                (eval_q_n, -1)
            )
        _, eval_k = eval_results.shape
        print(
            "Loaded eval queries", eval_queries.shape, "and results", eval_results.shape
        )
    for i, strat in enumerate(it):
        # block_until_ready() for more accurate progress.
        l0_graph = optimize_l0_graph_batched(
            graph=l0_graph,
            vecs=vecs,
            ef=ef,
            r=r,
            batch=batch or n,
            strat=strat,
            rk=l0_rks[i],
        ).block_until_ready()
        if pb:
            pb.update(1)
        if args.nn_samp:
            correct = (
                intersect_count_batch(nn_samp_knn, l0_graph[nn_samp_nodes]).sum().item()
            )
            correct_ratio = correct / (args.nn_samp * m_max)
            print(f"Iteration {i}: {correct_ratio * 100:.2f}% true k-NN found")
        if args.eval_q:
            # Find entry node at level 0. We drop straight to level 0 by simply doing a brute-force search over all level nodes.
            start_cands = level_graphs[0]["nodes"]  # (c,)
            # (qn, 1, d) - (1, c, d) = (qn, c, d)
            dist_to_start_cands = norm(
                eval_queries[:, None] - select_vecs(vecs, start_cands)[None], axis=2
            )  # (qn, c)
            starts = start_cands[np.argmin(dist_to_start_cands, axis=1)]  # (qn,)
            # Perform prune to reproduce how the final graph would be for accurate results.
            eval_graph = compute_robust_prune_batched(
                graph=l0_graph,
                vecs=vecs,
                m=m,
                dist_thresh=dist_thresh,
                batch=batch or n,
            )
            # Do greedy search.
            # NOTE: This isn't how our Vamana GreedySearch works (because we haven't merged the higher-level edges so it can't take those paths),
            # but is how HNSW search works (i.e. only within level 0), so it's still a good eval.
            eval_r_got = greedy_search(
                graph=eval_graph,
                vecs=vecs,
                target_vecs=eval_queries,
                k=eval_k,
                ef=eval_k * 2,  # Typically, ef is double k.
                iterations=eval_k,
                id_start=starts,
            )  # (qn, eval_k)
            correct = intersect_count_batch(eval_results, eval_r_got).sum().item()
            correct_ratio = correct / (eval_q_n * eval_k)
            print(f"Iteration {i}: {correct_ratio * 100:.2f}% correct query results")
    if pb:
        pb.close()

    # RobustPrune isn't just for accuracy, it also removes redundant edges for faster query times.
    print("Final prune")
    l0_graph = compute_robust_prune_batched(
        graph=l0_graph,
        vecs=vecs,
        m=m,
        dist_thresh=dist_thresh,
        batch=batch or n,
    ).block_until_ready()

    print("Saving", l0_graph.dtype, l0_graph.shape)
    with open(args.out, "wb") as f:
        # WARNING: Do not convert to Python type and serialize as MessagePack/JSON/etc. as that conversion + serialization process will be extremely slow in Python.
        f.write(l0_graph.tobytes())
    with open(args.out_levels, "wb") as f:
        f.write(
            msgpack.packb(
                [
                    {
                        "level": i + 1,
                        "nodes": level["nodes"].tobytes(),
                        "neighbors_for_each_node": level[
                            "neighbors_for_each_node"
                        ].tobytes(),
                    }
                    for i, level in enumerate(level_graphs)
                ]
            )  # type: ignore
        )
    with open(args.out_medoid, "w") as f:
        f.write(str(entry_node.item()))
    print("All done!")


if __name__ == "__main__":
    main()
