"""
Exercise 1 - Point Clouds: Geometry and Spread in 2D
ANN-DL 2026.2 - Exercise 1 (Data)
"""
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Class parameters (fixed for the whole exercise)
# ---------------------------------------------------------------------------
MEANS = {
    0: np.array([2.0, 3.0]),
    1: np.array([5.0, 6.0]),
    2: np.array([8.0, 1.0]),
    3: np.array([15.0, 4.0]),
}
STDS = {
    0: np.array([0.8, 2.5]),
    1: np.array([1.2, 1.9]),
    2: np.array([0.9, 0.9]),
    3: np.array([0.5, 2.0]),
}
N_PER_CLASS = 100
COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def generate_clouds(scale: float):
    """Generate the 4 classes with stds multiplied by `scale`. Means never change."""
    X, y = [], []
    for c in range(4):
        pts = rng.normal(loc=MEANS[c], scale=STDS[c] * scale, size=(N_PER_CLASS, 2))
        X.append(pts)
        y.append(np.full(N_PER_CLASS, c))
    return np.vstack(X), np.concatenate(y)


# ---------------------------------------------------------------------------
# A - Generate the clouds (s = 1) and plot Figure 1
# ---------------------------------------------------------------------------
X1, y1 = generate_clouds(scale=1.0)

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(4):
    mask = y1 == c
    ax.scatter(X1[mask, 0], X1[mask, 1], s=18, color=COLORS[c], alpha=0.7, label=f"Class {c}")
    ax.scatter(*MEANS[c], color=COLORS[c], marker="X", s=200, edgecolor="black", linewidth=1.5, zorder=5)
ax.set_title("Figure 1 - Point clouds (s = 1) with class centers marked")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_clouds.png", dpi=150)
plt.close(fig)
print("Figure 1 saved.")

# ---------------------------------------------------------------------------
# B - Spread study: s in {0.5, 1.0, 2.0, 4.0}
# ---------------------------------------------------------------------------
scales = [0.5, 1.0, 2.0, 4.0]
datasets = {s: generate_clouds(s) for s in scales}

all_x = np.concatenate([datasets[s][0][:, 0] for s in scales])
all_y = np.concatenate([datasets[s][0][:, 1] for s in scales])
pad_x = 0.05 * (all_x.max() - all_x.min())
pad_y = 0.05 * (all_y.max() - all_y.min())
xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for ax, s in zip(axes, scales):
    Xs, ys = datasets[s]
    for c in range(4):
        mask = ys == c
        ax.scatter(Xs[mask, 0], Xs[mask, 1], s=12, color=COLORS[c], alpha=0.7)
    ax.set_title(f"s = {s}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
fig.suptitle("Figure 2 - Same 4 classes at four spread levels (shared axes)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_spread.png", dpi=150)
plt.close(fig)
print("Figure 2 saved.")

# --- Separation ratio r_ij at s = 1 (uses the true generating mu, sigma) ---
sigma_bar = {c: STDS[c].mean() for c in range(4)}
pairs = list(itertools.combinations(range(4), 2))
r_ij_s1 = {}
print("\nSeparation ratio r_ij at s = 1:")
print(f"{'pair':>8} | {'r_ij':>8}")
for i, j in pairs:
    dist = np.linalg.norm(MEANS[i] - MEANS[j])
    r = dist / (sigma_bar[i] + sigma_bar[j])
    r_ij_s1[(i, j)] = r
    print(f"({i},{j})   | {r:8.4f}")

smallest_pair = min(r_ij_s1, key=r_ij_s1.get)
smallest_r_s1 = r_ij_s1[smallest_pair]
smallest_r_s2 = smallest_r_s1 / 2.0  # r_ij scales with 1/s
print(f"\nSmallest r_ij at s=1: pair {smallest_pair} -> {smallest_r_s1:.4f}")
print(f"That same pair's r_ij at s=2 (r_ij(s=1)/2): {smallest_r_s2:.4f}")

# --- Mixing rate for each s (nearest-center check against the fixed means) ---
mean_matrix = np.vstack([MEANS[c] for c in range(4)])


def mixing_rate(X, y):
    dists = np.linalg.norm(X[:, None, :] - mean_matrix[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float(np.mean(nearest != y))


print("\nMixing rate per scale:")
mixing = {}
for s in scales:
    Xs, ys = datasets[s]
    mr = mixing_rate(Xs, ys)
    mixing[s] = mr
    print(f"s = {s:>4}: mixing rate = {mr:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(scales, [mixing[s] for s in scales], marker="o")
ax.set_title("Figure 3 - Mixing rate x scale factor s")
ax.set_xlabel("scale factor s")
ax.set_ylabel("mixing rate")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_mixing.png", dpi=150)
plt.close(fig)
print("Figure 3 saved.")

print("\nDone. Figures written to:", FIGURES_DIR)

# ---------------------------------------------------------------------------
# C.2 - Sketch: nearest-center (Voronoi) partition over the 4 known means.
# This is a geometric construction from the given centers, not a trained
# model - it approximates what a boundary "by nearest center" would look
# like, matching the same criterion used for the mixing rate above.
# Far-away dummy points are added just to bound the diagram to a finite
# region; only ridges between two real class centers are drawn.
# ---------------------------------------------------------------------------
from scipy.spatial import Voronoi

centers = np.vstack([MEANS[c] for c in range(4)])
big = 1000
cx, cy = centers[:, 0].mean(), centers[:, 1].mean()
dummies = np.array([
    [big, big], [big, -big], [-big, big], [-big, -big],
    [cx, big], [cx, -big], [big, cy], [-big, cy],
])
vor = Voronoi(np.vstack([centers, dummies]))

xlim = (X1[:, 0].min() - 1, X1[:, 0].max() + 1)
ylim = (X1[:, 1].min() - 1, X1[:, 1].max() + 1)

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(4):
    mask = y1 == c
    ax.scatter(X1[mask, 0], X1[mask, 1], s=14, color=COLORS[c], alpha=0.5)
    ax.scatter(*MEANS[c], color=COLORS[c], marker="X", s=200, edgecolor="black", linewidth=1.5, zorder=5)

for (p1, p2), ridge in zip(vor.ridge_points, vor.ridge_vertices):
    if p1 < 4 and p2 < 4:
        ridge = np.asarray(ridge)
        if np.all(ridge >= 0):
            v1, v2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], "k--", linewidth=1.5)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("Figure 1 (annotated) - Sketch of nearest-center boundaries")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1b_sketch.png", dpi=150)
plt.close(fig)
print("Figure 1b (sketch) saved.")