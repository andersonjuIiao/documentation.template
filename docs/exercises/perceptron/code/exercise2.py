"""Exercise 2 — Overlapping data: the case the perceptron cannot solve.

Reuses the Exercise 1 implementation unchanged, adding only the pocket
algorithm's best-so-far tracking (already built into Perceptron.fit
via track_pocket=True). Produces Figures 4-6.

Run from anywhere, e.g.:
    python docs/exercises/perceptron/code/exercise2.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perceptron import Perceptron

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# --- A: generate the data -----------------------------------------------
N_PER_CLASS = 1000

class0 = rng.multivariate_normal([3, 3], [[1.5, 0], [0, 1.5]], size=N_PER_CLASS)
class1 = rng.multivariate_normal([4, 4], [[1.5, 0], [0, 1.5]], size=N_PER_CLASS)

X = np.vstack([class0, class1])
y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(class0[:, 0], class0[:, 1], s=10, alpha=0.5, label="Class 0", color="tab:blue")
ax.scatter(class1[:, 0], class1[:, 1], s=10, alpha=0.5, label="Class 1", color="tab:orange")
ax.set_title("Figure 4 — Overlapping data")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4.png", dpi=150)
plt.close(fig)

# --- B: train with pocket tracking ---------------------------------------
model = Perceptron(n_features=2, rng=rng, eta=0.01)
model.fit(X, y, max_epochs=100, track_pocket=True)

n_epochs_run = len(model.history_acc)
final_acc = model.history_acc[-1]

print("=== Exercise 2 ===")
print(f"epochs run = {n_epochs_run}")
print(f"final w = {model.w}")
print(f"final b = {model.b}")
print(f"final accuracy = {final_acc:.4f}")
print(f"pocket w = {model.pocket_w}")
print(f"pocket b = {model.pocket_b}")
print(f"pocket accuracy = {model.pocket_acc:.4f}")
print(f"pocket epoch (first reached) = {model.pocket_epoch}")

# --- C: figures ------------------------------------------------------------
# Figure 5: both boundaries, misclassified points (w.r.t. FINAL weights) marked
final_preds = model.predict(X)
misclassified = final_preds != y

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[(y == 0) & ~misclassified, 0], X[(y == 0) & ~misclassified, 1],
           s=10, alpha=0.5, color="tab:blue", label="Class 0")
ax.scatter(X[(y == 1) & ~misclassified, 0], X[(y == 1) & ~misclassified, 1],
           s=10, alpha=0.5, color="tab:orange", label="Class 1")
ax.scatter(X[misclassified, 0], X[misclassified, 1],
           s=25, facecolors="none", edgecolors="red", linewidths=0.8,
           label="Misclassified (final)")

x1_line = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

x2_final = -(model.w[0] * x1_line + model.b) / model.w[1]
ax.plot(x1_line, x2_final, color="black", linewidth=1.5, label="Final boundary")

x2_pocket = -(model.pocket_w[0] * x1_line + model.pocket_b) / model.pocket_w[1]
ax.plot(x1_line, x2_pocket, color="purple", linewidth=1.5, linestyle="--",
        label="Pocket boundary")

ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
ax.set_title("Figure 5 — Final vs. pocket decision boundary")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5.png", dpi=150)
plt.close(fig)

# Figure 6: current-weights accuracy vs. pocket (best-so-far) accuracy, per epoch
fig, ax = plt.subplots(figsize=(6, 4))
epochs = range(1, n_epochs_run + 1)
ax.plot(epochs, model.history_acc, color="tab:red", alpha=0.8, label="Current weights")
ax.plot(epochs, model.pocket_history_acc, color="tab:purple", linewidth=2, label="Pocket (best-so-far)")
ax.axhline(0.73, color="gray", linestyle=":", linewidth=1, label="Best possible line (~73%)")
ax.set_title("Figure 6 — Accuracy per epoch: current vs. pocket")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6.png", dpi=150)
plt.close(fig)
