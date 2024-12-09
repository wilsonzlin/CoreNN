from argparse import ArgumentParser
from index import calc_approx_medoid
from index import init_random_graph
from index import optimize_graph
from index import optimize_graph_batch
from util import arange
from util import NULL_ID
import jax
import jax.numpy as np
import msgpack

parser = ArgumentParser(description="Build a Vamana graph using the GPU.")
parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
parser.add_argument("--out", type=str, help="Graph output path")
parser.add_argument("--out-medoid", type=str, help="Medoid output path")
parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
parser.add_argument("--dim", type=int, help="Vector dimensions")
parser.add_argument("--m", type=int, help="Degree bound")
parser.add_argument("--ef", type=int, help="Search list cap")
parser.add_argument("--iter", type=int, help="Search iterations")
parser.add_argument("--alpha", type=float, help="Distance threshold")
parser.add_argument("--batch", type=int, help="Update batch size", default=64)
parser.add_argument(
    "--profile", type=str, help="Profile a batch optimization to this directory"
)
args = parser.parse_args()

print("Loading vectors")
dtype = np.dtype(args.dtype)
with open(args.vectors, "rb") as f:
    # Cast to bfloat16 for faster calculations.
    # WARNING: Do not use float16, it only has a range of +/-65504, which means a max dist of +/-sqrt(65504) => +/-255 (as we use L2 norms) which is too small.
    vecs = (
        np.frombuffer(f.read(), dtype=dtype).reshape((-1, args.dim)).astype(np.bfloat16)
    )
n, _ = vecs.shape
assert n < NULL_ID
m = args.m
ef = args.ef
search_iter = args.iter or args.ef
update_batch_size = args.batch
dist_thresh = np.bfloat16(args.alpha)
seed = 0
medoid_sample_size = 10_000
print(f"{n=} {m=} {ef=} {search_iter=} {update_batch_size=} {dist_thresh=}")

print("Calculating approx. medoid")
medoid = calc_approx_medoid(
    vecs=vecs,
    sample_size=medoid_sample_size,
    seed=seed,
)
print("Initializing random graph")
graph = init_random_graph(
    n=n,
    m=m,
    seed=seed,
)
if args.profile:
    with jax.profiler.trace(args.profile, create_perfetto_link=True):
        print("Profiling")
        optimize_graph_batch(
            graph=graph,
            vecs=vecs,
            batch_nodes=arange(update_batch_size),
            id_medoid=medoid,
            m=m,
            ef=ef,
            dist_thresh=dist_thresh,
        ).block_until_ready()
        print("Computation complete")
else:
    graph = optimize_graph(
        graph=graph,
        vecs=vecs,
        id_medoid=medoid,
        m=m,
        ef=ef,
        search_iter=search_iter,
        dist_thresh=dist_thresh,
        update_batch_size=update_batch_size,
        seed=seed,
    ).block_until_ready()
    print("Saving")
    with open(args.out, "wb") as f:
        # WARNING: Do not .dump to f as that is unbuffered and extremely slow for large graphs and/or high-latency storage (e.g. NFS).
        f.write(
            msgpack.dumps(
                {
                    i: [n for n in nodes if n != NULL_ID]
                    for i, nodes in enumerate(graph.tolist())
                },
            )  # type: ignore
        )
    with open(args.out_medoid, "w") as f:
        f.write(str(medoid.item()))
    print("All done!")
