from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Tuple
from util import plot_data_series_as_lines
from util import plot_time_series
import msgpack
import os

ds = os.environ["DS"]
base_dir = f"dataset/{ds}/out"


@dataclass
class Variant:
    name: str
    out_neighbor_count_hist: List[Tuple[float, int]]
    edge_dist_hist: List[Tuple[float, int]]
    medoid_dist_hist: List[Tuple[float, int]]
    query_metrics_means: Dict[str, List[float]]


variants: List[Variant] = [
    Variant(**v)
    for v in msgpack.unpackb(
        open(f"{base_dir}/_agg.msgpack", "rb").read(), strict_map_key=False
    )
]
print("Loaded data")

d = f"{base_dir}/_graphs"
os.makedirs(d, exist_ok=True)
labels = [v.name for v in variants]

plot_data_series_as_lines(
    datasets=[v.out_neighbor_count_hist for v in variants],
    labels=labels,
    output_path=f"{d}/out_neighbors.webp",
    title="Out-neighbor counts",
    xlabel="Neighbors",
    ylabel="Nodes",
)
print("Plotted out-neighbors")

plot_data_series_as_lines(
    datasets=[v.edge_dist_hist for v in variants],
    labels=labels,
    output_path=f"{d}/edge_dists.webp",
    title="Edge distances",
    xlabel="Euclidean distance",
    ylabel="Edges",
)
print("Plotted edge dists")

plot_data_series_as_lines(
    datasets=[v.medoid_dist_hist for v in variants],
    labels=labels,
    output_path=f"{d}/medoid_dists.webp",
    title="Medoid distances",
    xlabel="Euclidean distance",
    ylabel="Neighbors",
)
print("Plotted medoid dists")

for col in variants[0].query_metrics_means.keys():
    plot_time_series(
        arrays=[v.query_metrics_means[col] for v in variants],
        labels=labels,
        average=False,
        output_path=f"{d}/query_{col}.webp",
        title=f"Query {col}",
        xlabel="Search iteration",
    )
    print("Plotted query", col)
