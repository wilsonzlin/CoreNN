from util import plot_distribution
import numpy as np
import os

dataset = os.environ["DS"]

with open(f"dataset/{dataset}/out/full/edge-dists.mat", "rb") as f:
    full_dists = np.frombuffer(f.read(), dtype=np.float32)
print("Loaded edge dists")

plot_distribution(
    data=full_dists,
    output_path=f"dataset/{dataset}/out/full_dists.webp",
    title="Fully-connected distances",
)
print("Plotted edge dists")
