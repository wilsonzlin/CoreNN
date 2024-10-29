import matplotlib.pyplot as plt
import msgpack

edge_dists_by_level = msgpack.unpack(
    open("out/hnsw/edge_dists_by_level.msgpack", "rb"), strict_map_key=False
)
print("Loaded edge dists by level")

levels = len(edge_dists_by_level)

fig, axes = plt.subplots(levels, 1, figsize=(5, 4 * levels))
fig.suptitle("Graph edge distances by level", fontsize=16, y=1.02)

axes_flat = axes.flatten()

for level, edge_dists in edge_dists_by_level.items():
    ax = axes_flat[level]
    dists = [v for d in edge_dists.values() for v in d.values()]
    ax.hist(dists, bins=150, alpha=0.7)
    ax.set_title(f"Level {level}\n(n={len(edge_dists)} edges={len(dists)})")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(
    "out/hnsw/edge_dists_by_level.webp", format="webp", bbox_inches="tight", dpi=150
)
print("Rendered plot")
