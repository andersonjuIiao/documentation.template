"""
Exercise 2 - Non-Linearity in Higher Dimensions
ANN-DL 2026.2 - Exercise 1 (Data)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# A - Dataset I: shifted Gaussians (5D)
# ---------------------------------------------------------------------------
mu_A = np.array([0, 0, 0, 0, 0])
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
N = 500

X_A = rng.multivariate_normal(mu_A, Sigma_A, size=N)
X_B = rng.multivariate_normal(mu_B, Sigma_B, size=N)
X_dataset1 = np.vstack([X_A, X_B])
y_dataset1 = np.concatenate([np.zeros(N), np.ones(N)])

# ---------------------------------------------------------------------------
# B - Dataset II: concentric shells (5D)
# ---------------------------------------------------------------------------
def sample_shell(n, radius_mean, radius_std, dim=5):
    v = rng.normal(size=(n, dim))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(loc=radius_mean, scale=radius_std, size=(n, 1))
    return rho * u

X_C = sample_shell(N, radius_mean=2.0, radius_std=0.4)  # core
X_D = sample_shell(N, radius_mean=5.0, radius_std=0.4)  # shell
X_dataset2 = np.vstack([X_C, X_D])
y_dataset2 = np.concatenate([np.zeros(N), np.ones(N)])

# ---------------------------------------------------------------------------
# C - Visualize and compare
# ---------------------------------------------------------------------------
# 1) PCA to 2D for each dataset + Figure 4 (side by side)
pca1 = PCA(n_components=2, random_state=42)
proj1 = pca1.fit_transform(X_dataset1)
pca2 = PCA(n_components=2, random_state=42)
proj2 = pca2.fit_transform(X_dataset2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for label in [0, 1]:
    mask = y_dataset1 == label
    axes[0].scatter(proj1[mask, 0], proj1[mask, 1], s=15, alpha=0.6, label=f"class {int(label)}")
axes[0].set_title("Dataset I (shifted Gaussians) - PCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

for label in [0, 1]:
    mask = y_dataset2 == label
    axes[1].scatter(proj2[mask, 0], proj2[mask, 1], s=15, alpha=0.6, label=f"class {int(label)}")
axes[1].set_title("Dataset II (concentric shells) - PCA")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

fig.suptitle("Figure 4 - PCA projection to 2D")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_pca.png", dpi=150)
plt.close(fig)
print("Figure 4 saved.")

# 2) Explained variance of first two components
ev1 = pca1.explained_variance_ratio_
ev2 = pca2.explained_variance_ratio_
print("\nExplained variance (PC1 + PC2):")
print(f"Dataset I : PC1={ev1[0]:.4f}, PC2={ev1[1]:.4f}, sum={ev1[0]+ev1[1]:.4f}")
print(f"Dataset II: PC1={ev2[0]:.4f}, PC2={ev2[1]:.4f}, sum={ev2[0]+ev2[1]:.4f}")

# 3) Distance between class centers in 5D + Figure 5 (radius histograms)
dist1 = np.linalg.norm(X_A.mean(axis=0) - X_B.mean(axis=0))
dist2 = np.linalg.norm(X_C.mean(axis=0) - X_D.mean(axis=0))
print(f"\nDistance between class centers (5D):")
print(f"Dataset I  ||mu_A - mu_B|| = {dist1:.4f}")
print(f"Dataset II ||mu_C - mu_D|| = {dist2:.4f}")

radius_A = np.linalg.norm(X_A, axis=1)
radius_B = np.linalg.norm(X_B, axis=1)
radius_C = np.linalg.norm(X_C, axis=1)
radius_D = np.linalg.norm(X_D, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(radius_A, bins=30, alpha=0.6, label="Class A")
axes[0].hist(radius_B, bins=30, alpha=0.6, label="Class B")
axes[0].set_title("Dataset I - ||x|| per class")
axes[0].set_xlabel("||x||")
axes[0].set_ylabel("count")
axes[0].legend()

axes[1].hist(radius_C, bins=30, alpha=0.6, label="Class C (core)")
axes[1].hist(radius_D, bins=30, alpha=0.6, label="Class D (shell)")
axes[1].set_title("Dataset II - ||x|| per class")
axes[1].set_xlabel("||x||")
axes[1].set_ylabel("count")
axes[1].legend()

fig.suptitle("Figure 5 - Radius histograms, both classes overlaid")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_radius_hist.png", dpi=150)
plt.close(fig)
print("Figure 5 saved.")

print("\nDone.")