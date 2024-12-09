from collections import defaultdict
from greedy_search_test import ref_greedy_search
from index import init_random_graph
from index import optimize_graph_batch
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from robust_prune_test import ref_compute_robust_pruned
from typing import Dict
from typing import List
from util import arange
from util import NULL_ID
import jax.numpy as np
import jax.random as rand


def ref_optimize_graph_batch(
    *,
    g: UInt32[Array, "n m"],
    vecs: Float16[Array, "n d"],
    batch_nodes: List[int],
    start: int,
    ef: int,
    search_iter: int,
    dist_thresh: Float16[Array, ""],
    degree_bound: int,
):
    ref_g = g.tolist()
    ref_add_edges: Dict[int, List[int]] = defaultdict(list)
    for n in batch_nodes:
        visited = ref_greedy_search(
            vecs=vecs,
            start=start,
            g=g,
            q=n,
            k=0,
            search_list_cap=ef,
            search_iter=search_iter,
        )[1]
        new_neighbors = ref_compute_robust_pruned(
            vecs=vecs,
            p=n,
            candidates=visited + ref_g[n],
            distance_threshold=dist_thresh,
            degree_bound=degree_bound,
        )
        ref_g[n] = new_neighbors
        for j in new_neighbors:
            ref_add_edges[j].append(n)
    for n, add in ref_add_edges.items():
        ref_g[n] = ref_compute_robust_pruned(
            vecs=vecs,
            p=n,
            candidates=ref_g[n] + add,
            distance_threshold=dist_thresh,
            degree_bound=degree_bound,
        )

    # Convert ref_g to matrix.
    for r in ref_g:
        l = len(r)
        assert l <= degree_bound
        if l < degree_bound:
            r.extend([NULL_ID] * (degree_bound - l))
    return np.array(ref_g, dtype=np.uint32)


def test_optimize_graph_batch():
    n = 15_000
    degree_bound = 20
    d = 128
    seed = 0
    dist_thresh = np.float16(1.1)
    ef = 80
    search_iter = 100
    batch_size = 10
    start = 0
    g = init_random_graph(n=n, m=degree_bound, seed=seed)
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    batch_nodes = arange(batch_size)

    exp = ref_optimize_graph_batch(
        g=g,
        vecs=vecs,
        batch_nodes=batch_nodes,
        start=start,
        ef=ef,
        search_iter=search_iter,
        dist_thresh=dist_thresh,
        degree_bound=degree_bound,
    )

    got = optimize_graph_batch(
        graph=g,
        vecs=vecs,
        batch_nodes=batch_nodes,
        id_medoid=start,
        m=degree_bound,
        ef=ef,
        search_iter=search_iter,
        dist_thresh=dist_thresh,
    )

    assert not np.array_equal(exp, g)
    assert np.array_equal(exp, got)
