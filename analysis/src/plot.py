from util import plot_distribution
from util import plot_time_series
import msgpack
import numpy as np
import sys

# Path to dir containing output data e.g. out/vamana.
d = sys.argv[1]

graph = msgpack.unpack(open(f"{d}/graph.msgpack", "rb"), strict_map_key=False)
print("Loaded graph")

edge_dists = msgpack.unpack(open(f"{d}/edge_dists.msgpack", "rb"), strict_map_key=False)
print("Loaded edge dists")

medoid_dists = np.frombuffer(
    open(f"{d}/medoid_dists.mat", "rb").read(), dtype=np.float32
)
print("Loaded medoid dists")

query_metrics = msgpack.unpack(
    open(f"{d}/query_metrics.msgpack", "rb"), strict_map_key=False
)
print("Loaded query metrics")

plot_distribution(
    data=[len(v) for v in graph.values()],
    output_path=f"{d}/out_neighbors.webp",
    title="Out-neighbor counts",
)
print("Plotted out-neighbors")

plot_distribution(
    data=[v for d in edge_dists.values() for v in d.values()],
    output_path=f"{d}/edge_dists.webp",
    title="ANN edge distances",
)
print("Plotted edge dists")

plot_distribution(
    data=medoid_dists,
    output_path=f"{d}/medoid_dists.webp",
    title="Medoid distances",
    xlabel="Euclidean distance",
    ylabel="Neighbors",
)
print("Plotted medoid dists")

plot_distribution(
    data=[len(q["iterations"]) for q in query_metrics],
    output_path=f"{d}/query_iterations.webp",
    title="Query iterations",
)
print("Plotted query iterations")

iters = query_metrics[0]["iterations"]
for col in iters[0].keys():
    plot_time_series(
        arrays=[[i[col] for i in q["iterations"]] for q in query_metrics],
        output_path=f"{d}/query_{col}.webp",
        title=f"Query {col}",
        xlabel="Search iteration",
    )
    print("Plotted query", col)
