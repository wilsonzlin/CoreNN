# DO NOT OPTIMIZE WITHOUT PROFILING FIRST.
# Use this script to help you find what's actually the bottleneck,
# instead of burning days going down rabbit holes chasing red herrings.
# Add more benchmarks here, instead of guessing/assuming.
# WARNING: Make sure to return from bench_*, or else the JIT will optimize away the function.

from argparse import ArgumentParser
from greedy_search import greedy_search_ids
from index import init_random_graph
from jax.numpy.linalg import norm
from robust_prune import compute_robust_pruned
from util import arange
from util import group_by_y_id
from util import NULL_ID
import jax
import jax.numpy as np
import jax.random as rand
import time

parser = ArgumentParser(description="Build a Vamana graph using the GPU.")
parser.add_argument("--n", type=int, help="Vectors")
parser.add_argument("--dim", type=int, help="Vector dimensions", default=128)
parser.add_argument("--m", type=int, help="Degree bound")
parser.add_argument("--ef", type=int, help="Search list cap")
parser.add_argument("--iter", type=int, help="Search iterations (defaults to `ef`)")
parser.add_argument("--b-search", type=int, help="Search batch size", default=64)
parser.add_argument("--b-update", type=int, help="Update batch size", default=64)
parser.add_argument("--cand", type=int, help="Prune candidates (defaults to `iter`)")
args = parser.parse_args()

seed = 0
rk = rand.PRNGKey(seed)
n = args.n
assert n < NULL_ID
dim = args.dim
vecs = rand.uniform(rk, (n, dim), dtype=np.bfloat16)
m = args.m
ef = args.ef
search_iter = args.iter or ef
b_search = args.b_search
b_update = args.b_update
cand = args.cand or (m + search_iter)
dist_thresh = np.bfloat16(1.1)  # No effect on performance.
print(f"{n=} {m=} {ef=} {search_iter=} {b_search=} {b_update=} {cand=}")
print("=============")

bench_warmups = 3
bench_iterations = 200

# Prebuild these so their times doesn't affect benchmarking.
# The actual values shouldn't affect computation time so using the same values for all benchmarks should be OK.
cand_ids = rand.choice(rk, arange(n), shape=(b_update, cand))
search_node_ids = arange(b_search)
update_node_ids = arange(b_update)
# group_by_y_id is used during the backedge insertion process, which is why these shapes are used.
group_by_y_id_xs = rand.choice(rk, arange(n), shape=(b_update,))
group_by_y_id_ys = rand.choice(rk, arange(n), shape=(b_update, m))
search_argsort_vals = rand.uniform(rk, (b_search, ef + m), dtype=np.bfloat16)
norm_vecs = vecs[:b_search]

graph = init_random_graph(
    n=n,
    m=m,
    seed=seed,
)


def bench_greedy_search():
    return greedy_search_ids(
        graph=graph,
        vecs=vecs,
        id_targets=search_node_ids,
        k=None,
        ef=ef,
        iterations=search_iter,
        id_start=0,
    )


def bench_compute_robust_pruned():
    return compute_robust_pruned(
        cand_ids=cand_ids,
        dist_thresh=dist_thresh,
        m=m,
        node_ids=update_node_ids,
        vecs=vecs,
    )


def bench_group_by_y_id():
    return group_by_y_id(
        M_id=group_by_y_id_xs,
        M=group_by_y_id_ys,
    )[1]


@jax.jit
def bench_search_argsort():
    return np.argsort(search_argsort_vals, axis=1)


@jax.jit
def bench_search_argpartition():
    return np.argpartition(search_argsort_vals, ef // 2, axis=1)


@jax.jit
def bench_search_argmin():
    return np.argmin(search_argsort_vals, axis=1)


@jax.jit
def bench_search_norm_pairwise():
    return norm(norm_vecs[:, None, :] - norm_vecs[None, :, :], axis=2)


benches = {
    "greedy_search": bench_greedy_search,
    "compute_robust_pruned": bench_compute_robust_pruned,
    "group_by_y_id": bench_group_by_y_id,
    "search_argsort": bench_search_argsort,
    "search_argpartition": bench_search_argpartition,
    "search_argmin": bench_search_argmin,
    "search_norm_pairwise": bench_search_norm_pairwise,
}

for name, bench in benches.items():
    for _ in range(bench_warmups):
        # Warm up.
        bench()
    started = time.perf_counter()
    for _ in range(bench_iterations):
        bench().block_until_ready()
    elapsed = time.perf_counter() - started
    print(f"{name} ×{bench_iterations}: {elapsed:.2f}s")
