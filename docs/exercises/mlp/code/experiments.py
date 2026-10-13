"""Exercises 2, 3 and 4: data, training, figures and numbers.

Run from the repository root (or anywhere):
    python docs/exercises/mlp/code/experiments.py

A single rng = np.random.default_rng(42) is created once and shared by every
model below, in the order they appear. The only extra generators are the
per-seed ones of the Exercise 4 multi-seed comparison, which the statement
explicitly asks for (at least 3 different seeds).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP

rng = np.random.default_rng(42)  # the one shared generator
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPOCHS = 2000
LR = 0.5


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


# =====================================================================
# Exercise 2 - binary classification a straight line cannot do
# =====================================================================
print("=" * 60, "\nExercise 2\n" + "=" * 60)

# A - data (parameters unchanged from the statement)
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X2, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                   cluster_std=1.2, random_state=42)
y2 = (c > 0).astype(int)  # class 0: 1 cluster | class 1: 2 clusters
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)
print(f"train: {len(y2_tr)}  test: {len(y2_te)}  "
      f"class balance (train): {np.bincount(y2_tr)}")

fig, ax = plt.subplots(figsize=(6, 5))
for k, color in [(0, "tab:blue"), (1, "tab:orange")]:
    ax.scatter(*X2[y2 == k].T, s=10, alpha=0.6, c=color, label=f"Class {k}")
ax.set(title="Figure 1 - Exercise 2 data (1000 points)",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend()
save(fig, "fig1.png")

# B - linear baseline (sklearn allowed here: it is not our model)
baseline2 = LogisticRegression().fit(X2_tr, y2_tr).score(X2_te, y2_te)
print(f"Linear baseline test accuracy: {baseline2:.4f}")

# C - our MLP
arch2 = [2, 8, 1]
mlp2 = MLP(arch2, output="sigmoid", rng=rng)
hist2 = mlp2.fit(X2_tr, y2_tr.reshape(-1, 1), epochs=EPOCHS, lr=LR)
acc2 = (mlp2.predict(X2_te) == y2_te).mean()
print(f"MLP {arch2}, lr={LR}, epochs={EPOCHS}")
print(f"  initial train loss: {hist2[0]:.4f}  final train loss: {hist2[-1]:.4f}")
print(f"  test accuracy: {acc2:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist2, label=f"MLP {arch2}")
ax.set(title="Figure 2 - Exercise 2 training loss",
       xlabel="Epoch", ylabel="Binary cross-entropy (train)")
ax.legend()
save(fig, "fig2.png")

# Decision boundary: evaluate the network on a grid and shade the regions
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
gx, gy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_pred = mlp2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(gx, gy, grid_pred, levels=[-0.5, 0.5, 1.5],
            colors=["tab:blue", "tab:orange"], alpha=0.25)
for k, color in [(0, "tab:blue"), (1, "tab:orange")]:
    ax.scatter(*X2_te[y2_te == k].T, s=14, c=color, edgecolors="k",
               linewidths=0.3, label=f"Class {k} (test)")
ax.set(title="Figure 3 - Exercise 2 decision boundary (test set)",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend()
save(fig, "fig3.png")

# =====================================================================
# Exercise 3 - multi-class, same network
# =====================================================================
print("=" * 60, "\nExercise 3\n" + "=" * 60)

X3, y3 = make_classification(n_samples=1500, n_features=4, n_informative=4,
                             n_redundant=0, n_repeated=0, n_classes=3,
                             n_clusters_per_class=2, class_sep=1.0,
                             random_state=42)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3)
Y3_tr = np.eye(3)[y3_tr]  # one-hot targets for softmax + cross-entropy
print(f"train: {len(y3_tr)}  test: {len(y3_te)}")

baseline3 = LogisticRegression().fit(X3_tr, y3_tr).score(X3_te, y3_te)
print(f"Logistic-regression baseline test accuracy: {baseline3:.4f}")

# Same MLP class: only the arguments change (output size, output type)
arch3 = [4, 16, 3]
mlp3 = MLP(arch3, output="softmax", rng=rng)
hist3 = mlp3.fit(X3_tr, Y3_tr, epochs=EPOCHS, lr=LR)
pred3 = mlp3.predict(X3_te)
acc3 = (pred3 == y3_te).mean()
print(f"MLP {arch3}, lr={LR}, epochs={EPOCHS}, params={mlp3.n_params()}")
print(f"  initial train loss: {hist3[0]:.4f}  final train loss: {hist3[-1]:.4f}")
print(f"  test accuracy: {acc3:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist3, label=f"MLP {arch3}")
ax.set(title="Figure 4 - Exercise 3 training loss",
       xlabel="Epoch", ylabel="Categorical cross-entropy (train)")
ax.legend()
save(fig, "fig4.png")

cm = confusion_matrix(y3_te, pred3)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)
off = cm + cm.T
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most-confused pair: classes {i} and {j} "
      f"({cm[i, j]} + {cm[j, i]} = {off[i, j]} test errors)")

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(cm, cmap="Blues")
for r in range(3):
    for s in range(3):
        ax.text(s, r, cm[r, s], ha="center", va="center",
                color="white" if cm[r, s] > cm.max() / 2 else "black")
ax.set_xticks(range(3), [f"Class {k}" for k in range(3)])
ax.set_yticks(range(3), [f"Class {k}" for k in range(3)])
ax.set(title="Figure 5 - Exercise 3 confusion matrix (test)",
       xlabel="Predicted class", ylabel="True class")
fig.colorbar(im, ax=ax, label="Number of test samples")
save(fig, "fig5.png")

# =====================================================================
# Exercise 4 - does a second hidden layer help?
# =====================================================================
print("=" * 60, "\nExercise 4\n" + "=" * 60)

# A - deeper network, same data, same epochs, same learning rate, same rng
arch4 = [4, 16, 16, 3]
mlp4 = MLP(arch4, output="softmax", rng=rng)
hist4 = mlp4.fit(X3_tr, Y3_tr, epochs=EPOCHS, lr=LR)
acc4 = (mlp4.predict(X3_te) == y3_te).mean()
print(f"MLP {arch4}, lr={LR}, epochs={EPOCHS}, params={mlp4.n_params()}")
print(f"  final train loss: {hist4[-1]:.4f}  test accuracy: {acc4:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist3, label=f"Exercise 3: {arch3} (1 hidden layer)")
ax.plot(hist4, label=f"Exercise 4: {arch4} (2 hidden layers)")
ax.set(title="Figure 6 - Training loss, 1 vs 2 hidden layers",
       xlabel="Epoch", ylabel="Categorical cross-entropy (train)")
ax.legend()
save(fig, "fig6.png")

# B - controlled comparison over several seeds (only the init changes)
SEEDS = [42, 43, 44]
results = {"Ex3": [], "Ex4": []}
for s in SEEDS:
    for name, arch in [("Ex3", arch3), ("Ex4", arch4)]:
        m = MLP(arch, output="softmax", rng=np.random.default_rng(s))
        h = m.fit(X3_tr, Y3_tr, epochs=EPOCHS, lr=LR)
        acc = (m.predict(X3_te) == y3_te).mean()
        results[name].append((h[-1], acc))
        print(f"  seed {s} {name} {arch}: final train loss {h[-1]:.4f}, "
              f"test acc {acc:.4f}")

print("\nComparison (mean +- std over seeds", SEEDS, ")")
print(f"{'':6s}{'hidden':>8s}{'params':>8s}{'train loss':>18s}{'test acc':>18s}")
for name, arch, model in [("Ex3", arch3, mlp3), ("Ex4", arch4, mlp4)]:
    r = np.array(results[name])
    print(f"{name:6s}{len(arch) - 2:>8d}{model.n_params():>8d}"
          f"{r[:, 0].mean():>11.4f} +- {r[:, 0].std():.4f}"
          f"{r[:, 1].mean():>11.4f} +- {r[:, 1].std():.4f}")
diff = (np.array(results["Ex4"])[:, 1] - np.array(results["Ex3"])[:, 1])
print(f"Accuracy difference Ex4 - Ex3 (mean over seeds): {diff.mean():+.4f}")
