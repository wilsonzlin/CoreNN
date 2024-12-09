from greedy_search import greedy_search
from index import init_random_graph
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from typing import List
from typing import Tuple
import jax.numpy as np
import jax.random as rand


# Reference implementation of GreedySearch, according to the paper. Intentionally unoptimized to keep as close to correct reference code as possible.
# NOTE: It's expected that our greedy_search is close to this, but not 100%, as our implementation has slight differences.
def ref_greedy_search(
    *,
    vecs: Float16[Array, "n d"],
    start: int,
    g: UInt32[Array, "n m"],
    q: int,
    k: int,
    search_list_cap: int,
):
    search_list: List[Tuple[int, float]] = []
    search_list.append((start, norm(vecs[q] - vecs[start])))
    visited: List[int] = []
    while True:
        p_star = next(filter(lambda n: n[0] not in visited, search_list), None)
        if p_star is None:
            break
        p_star_id, _ = p_star
        for n in g[p_star_id].tolist():
            if not any(o == n for o, _ in search_list):
                search_list.append((n, norm(vecs[q] - vecs[n])))
        visited.append(p_star_id)
        search_list.sort(key=lambda o: o[1])
        del search_list[search_list_cap:]
    return search_list[:k], visited


def test_greedy_search_top_k():
    n = 15_000
    degree_bound = 12
    d = 128
    search_list_cap = 100
    k = 10
    start = 0
    query = 7777
    seed = 0
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    g = init_random_graph(n=n, m=degree_bound, seed=seed)

    exp = ref_greedy_search(
        vecs=vecs, start=start, g=g, q=query, k=k, search_list_cap=search_list_cap
    )
    exp_ids = {o for o, _ in exp[0]}
    assert len(exp_ids) == k
    got_ids = set(
        greedy_search(
            vecs=vecs,
            id_start=start,
            graph=g,
            id_targets=np.array([query]),
            k=k,
            ef=search_list_cap,
            iterations=search_list_cap,
        )
        .squeeze()
        .tolist()
    )
    assert len(got_ids) == k
    accuracy = len(exp_ids.intersection(got_ids)) / k
    print(f"Top-k accuracy: {accuracy:.2%}")


def test_greedy_search_visited():
    n = 15_000
    degree_bound = 12
    d = 128
    search_list_cap = 100
    start = 0
    query = 7777
    seed = 0
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    g = init_random_graph(n=n, m=degree_bound, seed=seed)

    exp = ref_greedy_search(
        vecs=vecs, start=start, g=g, q=query, k=0, search_list_cap=search_list_cap
    )[1]
    got = (
        greedy_search(
            vecs=vecs,
            id_start=start,
            graph=g,
            id_targets=np.array([query]),
            k=None,
            ef=search_list_cap,
            iterations=search_list_cap,
        )
        .squeeze()
        .tolist()
    )
    assert got == exp[: len(got)]
    print(len(exp) - len(got), "extra nodes visited by reference implementation")
