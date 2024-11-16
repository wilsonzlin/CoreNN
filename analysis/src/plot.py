from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict
from typing import List
from util import plot_distributions_as_lines
from util import plot_time_series
import msgpack
import numpy as np
import numpy.typing as npt
import os

ds = os.environ["DS"]
base_dir = f"dataset/{ds}/out"


@dataclass
class Variant:
    name: str
    out_neighbor_counts: npt.NDArray[np.uint16]
    edge_dists: npt.NDArray[np.float32]
    medoid_dists: npt.NDArray[np.float32]
    query_metrics_means: Dict[str, npt.NDArray[np.float32]]


variants: List[Variant] = []
for variant in tqdm(os.listdir(base_dir)):
    if variant.startswith("_"):
        continue

    d = f"{base_dir}/{variant}"

    graph = msgpack.unpack(open(f"{d}/graph.msgpack", "rb"), strict_map_key=False)

    edge_dists_raw = msgpack.unpack(
        open(f"{d}/edge_dists.msgpack", "rb"), strict_map_key=False
    )

    medoid_dists = np.frombuffer(
        open(f"{d}/medoid_dists.mat", "rb").read(), dtype=np.float32
    )

    query_metrics = msgpack.unpack(
        open(f"{d}/query_metrics.msgpack", "rb"), strict_map_key=False
    )

    out_neighbor_counts = np.array([len(v) for v in graph.values()], dtype=np.uint16)
    edge_dists = np.array(
        [v for d in edge_dists_raw.values() for v in d.values()], dtype=np.float32
    )
    query_metrics_means = {}
    # query_metrics: an array of { iterations: Array<{ [metric_name: string]: number }> }, one for each query.
    for metric_name in query_metrics[0]["iterations"][0].keys():
        arrays = [
            np.array([it[metric_name] for it in q["iterations"]]) for q in query_metrics
        ]
        max_len = max(len(a) for a in arrays)
        # Cast to float64 to support averaging.
        # Pad with NaNs so we can use nanmean.
        padded = [
            np.pad(a.astype(np.float64), (0, max_len - len(a)), constant_values=np.nan)
            for a in arrays
        ]
        stacked = np.vstack(padded, dtype=np.float32)
        mean = np.nanmean(stacked, axis=0)
        query_metrics_means[metric_name] = mean

    variants.append(
        Variant(
            name=variant,
            out_neighbor_counts=out_neighbor_counts,
            edge_dists=edge_dists,
            medoid_dists=medoid_dists,
            query_metrics_means=query_metrics_means,
        )
    )

d = f"{base_dir}/_graphs"
os.makedirs(d, exist_ok=True)
labels = [v.name for v in variants]

plot_distributions_as_lines(
    datasets=[v.out_neighbor_counts for v in variants],
    labels=labels,
    output_path=f"{d}/out_neighbors.webp",
    title="Out-neighbor counts",
    xlabel="Neighbors",
    ylabel="Nodes",
)
print("Plotted out-neighbors")

plot_distributions_as_lines(
    datasets=[v.edge_dists for v in variants],
    labels=labels,
    output_path=f"{d}/edge_dists.webp",
    title="Edge distances",
    xlabel="Euclidean distance",
    ylabel="Edges",
)
print("Plotted edge dists")

plot_distributions_as_lines(
    datasets=[v.medoid_dists for v in variants],
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
