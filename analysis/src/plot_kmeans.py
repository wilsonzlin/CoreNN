from sklearn.cluster import MiniBatchKMeans
from util import read_vectors
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import os

dataset = os.environ["DS"]

edge_dists = msgpack.unpack(
    open(f"dataset/{dataset}/out/vamana/edge_dists.msgpack", "rb"), strict_map_key=False
)
print("Loaded edge dists")

vecs = read_vectors("dataset/{dataset}/base.fvecs", np.float32)
print("Loaded vectors")

k = 32
kmeans = MiniBatchKMeans(n_clusters=k)
kmeans.fit(vecs)
cluster_labels = kmeans.predict(vecs)
print("Calculated k-means")

columns = 4
rows = k // columns

fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
fig.suptitle("Graph edge distances by cluster", fontsize=16, y=1.02)

axes_flat = axes.flatten()


all_values = np.array([v for d in edge_dists.values() for v in d.values()])
# Calculate global min and max for x-axis
global_min = np.min(all_values)
global_max = np.max(all_values)
# Calculate common bins for all histograms
all_bins = np.linspace(global_min, global_max, 150)
# Find the maximum frequency across all clusters for y-axis normalization
max_frequency = 0
for i in range(k):
    cluster_data = []
    for src, dists in edge_dists.items():
        cluster = cluster_labels[src]
        if cluster == i:
            cluster_data.extend(dists.values())
    if len(cluster_data) > 0:
        hist, _ = np.histogram(cluster_data, bins=all_bins)
        max_frequency = max(max_frequency, max(hist))

# Create histograms for each cluster
for i in range(k):
    ax = axes_flat[i]
    cluster_data = []
    for src, dists in edge_dists.items():
        cluster = cluster_labels[src]
        if cluster == i:
            cluster_data.extend(dists.values())

    if len(cluster_data) > 0:
        ax.hist(cluster_data, bins=all_bins, alpha=0.7)
        ax.set_title(f"Cluster {i}\n(n={len(cluster_data)})")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(0, max_frequency * 1.1)  # Add 10% padding to y-axis
    print(f"Plotted cluster {i}")

plt.tight_layout()
plt.savefig("out/edge_dists.kmeans.webp", format="webp", bbox_inches="tight", dpi=150)
print("Rendered plot")


# Calculate pairwise Euclidean distances between cluster centers
cluster_centers = kmeans.cluster_centers_
pairwise_dists = np.linalg.norm(
    cluster_centers[:, np.newaxis] - cluster_centers, axis=2
)

# Ignore the diagonal by setting it to NaN
np.fill_diagonal(pairwise_dists, np.nan)

# Plot the pairwise distances matrix
plt.figure(figsize=(8, 6))
plt.imshow(pairwise_dists, cmap="RdYlGn_r", interpolation="nearest")
plt.colorbar(label="Euclidean distance")
plt.title("Pairwise Euclidean distances between k-means clusters")
plt.xlabel("Cluster index")
plt.ylabel("Cluster index")
plt.xticks(range(k))
plt.yticks(range(k))
plt.savefig("out/kmeans_dists.matrix.webp", format="webp", bbox_inches="tight", dpi=150)
print("Rendered k-means cluster pairwise distances matrix")

# Render a histogram of pairwise cluster distances
pairwise_dists_flat = pairwise_dists[~np.isnan(pairwise_dists)]
plt.figure(figsize=(10, 6))
plt.hist(pairwise_dists_flat, bins=30, alpha=0.7)
plt.title("Histogram of pairwise Euclidean distances between k-means clusters")
plt.xlabel("Euclidean distance")
plt.ylabel("Frequency")
plt.savefig(
    "out/kmeans_dists.histogram.webp", format="webp", bbox_inches="tight", dpi=150
)
print("Rendered histogram of pairwise cluster distances")
