"""Exercise 1 — Separable data: the case the perceptron was designed for.

Generates the linearly separable dataset, trains the perceptron
(item B/C), and produces Figures 1-3. Reruns training with eta=1.0
for item D2.

Run from anywhere, e.g.:
    python docs/exercises/perceptron/code/exercise1.py
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

class0 = rng.multivariate_normal([1.5, 1.5], [[0.5, 0], [0, 0.5]], size=N_PER_CLASS)
class1 = rng.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], size=N_PER_CLASS)

X = np.vstack([class0, class1])
y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(class0[:, 0], class0[:, 1], s=10, alpha=0.6, label="Class 0", color="tab:blue")
ax.scatter(class1[:, 0], class1[:, 1], s=10, alpha=0.6, label="Class 1", color="tab:orange")
ax.set_title("Figure 1 — Separable data")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1.png", dpi=150)
plt.close(fig)

# --- B/C: train with eta = 0.01 ------------------------------------------
model = Perceptron(n_features=2, rng=rng, eta=0.01)
model.fit(X, y, max_epochs=100)

final_epochs = len(model.history_acc)
final_acc = model.history_acc[-1]

print("=== Exercise 1 — eta = 0.01 ===")
print(f"w = {model.w}")
print(f"b = {model.b}")
print(f"epochs to convergence = {final_epochs}")
print(f"final accuracy = {final_acc:.4f}")

# Figure 2: decision boundary + misclassified points
preds = model.predict(X)
misclassified = preds != y

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[(y == 0) & ~misclassified, 0], X[(y == 0) & ~misclassified, 1],
           s=10, alpha=0.6, color="tab:blue", label="Class 0")
ax.scatter(X[(y == 1) & ~misclassified, 0], X[(y == 1) & ~misclassified, 1],
           s=10, alpha=0.6, color="tab:orange", label="Class 1")
ax.scatter(X[misclassified, 0], X[misclassified, 1],
           s=40, facecolors="none", edgecolors="red", linewidths=1.5,
           label="Misclassified")

x1_line = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
x2_line = -(model.w[0] * x1_line + model.b) / model.w[1]
ax.plot(x1_line, x2_line, color="black", linewidth=1.5, label="Decision boundary")
ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

ax.set_title("Figure 2 — Decision boundary (eta = 0.01)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2.png", dpi=150)
plt.close(fig)

# Figure 3: accuracy x epoch
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, final_epochs + 1), model.history_acc, marker="o", markersize=3,
        color="tab:green", label="Accuracy (eta = 0.01)")
ax.set_title("Figure 3 — Accuracy per epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3.png", dpi=150)
plt.close(fig)

# --- D2: rerun with eta = 1.0, same rng stream, nothing else changed -----
model_eta1 = Perceptron(n_features=2, rng=rng, eta=1.0)
model_eta1.fit(X, y, max_epochs=100)

epochs_eta1 = len(model_eta1.history_acc)
acc_eta1 = model_eta1.history_acc[-1]

dir_001 = model.w / np.linalg.norm(model.w)
dir_1 = model_eta1.w / np.linalg.norm(model_eta1.w)

print("\n=== Exercise 1 — eta = 1.0 (item D2) ===")
print(f"w = {model_eta1.w}")
print(f"b = {model_eta1.b}")
print(f"epochs to convergence = {epochs_eta1}")
print(f"final accuracy = {acc_eta1:.4f}")
print(f"direction w/||w|| (eta=0.01) = {dir_001}")
print(f"direction w/||w|| (eta=1.0)  = {dir_1}")
print(f"cosine similarity between the two directions = {np.dot(dir_001, dir_1):.6f}")
