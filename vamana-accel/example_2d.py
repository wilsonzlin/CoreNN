from argparse import ArgumentParser
from index import calc_approx_medoid
from index import init_random_graph
from index import optimize_graph
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from util import NULL_ID
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt

parser = ArgumentParser(description="Visualize using random 2D vectors")
parser.add_argument("--n", type=int, help="Vectors", default=1000)
parser.add_argument("--m", type=int, help="Degree bound", default=10)
parser.add_argument("--ef", type=int, help="Search list cap", default=20)
parser.add_argument("--iter", type=int, help="Search iterations", default=20)
parser.add_argument("--alpha", type=float, help="Distance threshold", default=1.1)
parser.add_argument("--batch", type=int, help="Update batch size", default=8)
parser.add_argument("--seed", type=int, help="Random seed", default=0)
args = parser.parse_args()

n = args.n
assert n < NULL_ID
seed = args.seed
vecs = rand.uniform(rand.PRNGKey(seed), (n, 2), dtype=np.float16, minval=-1, maxval=1)
m = args.m
ef = args.ef
search_iter = args.iter
update_batch_size = args.batch
dist_thresh = np.float16(args.alpha)

medoid = calc_approx_medoid(
    vecs=vecs,
    sample_size=n,
    seed=seed,
)
init_graph = init_random_graph(
    n=n,
    m=m,
    seed=seed,
)
graph = optimize_graph(
    graph=init_graph,
    vecs=vecs,
    id_medoid=medoid,
    m=m,
    ef=ef,
    search_iter=search_iter,
    dist_thresh=dist_thresh,
    update_batch_size=update_batch_size,
    seed=seed,
).block_until_ready()


def plot_graph(
    points: Float16[Array, "n 2"], edges: UInt32[Array, "n m"], filename: str
):
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=1, c="black")

    for i, row in enumerate(edges):
        for j in row:
            if j != NULL_ID:
                plt.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="gray",
                    alpha=0.1,
                    linewidth=0.5,
                )

    plt.axis("off")
    plt.savefig(filename, format="webp", dpi=300, bbox_inches="tight")
    plt.close()


plot_graph(vecs, init_graph, "example_2d.init_graph.webp")
plot_graph(vecs, graph, "example_2d.graph.webp")
