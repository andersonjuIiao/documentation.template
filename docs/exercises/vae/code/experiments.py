"""Exercises 1 and 2: data, training of both VAEs, figures and numbers.

Run from the repository root:
    python docs/exercises/vae/code/experiments.py

MNIST is downloaded to ./data (repository root, ignored by git).
Seeds: torch.manual_seed(42) right before building each model, and the
split generator given in the statement. Figure 3 uses torch.manual_seed(0).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import random_split
from torchvision import datasets, transforms

from vae import VAE, run_epoch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
BETA = 1.0


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


# =====================================================================
# Exercise 1 - A: load and split the data (exactly as in the statement)
# =====================================================================
train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
train, val = random_split(train_full, [50_000, 10_000],
                          generator=torch.Generator().manual_seed(42))


def to_matrix(dataset, indices=None):
    """Stack images as flattened 784-vectors in [0, 1], plus their labels.

    Dividing the uint8 pixels by 255 is exactly what ToTensor() does; reading
    the tensors directly is just much faster than going image by image.
    """
    data, targets = dataset.data, dataset.targets
    if indices is not None:
        data, targets = data[indices], targets[indices]
    return data.reshape(-1, 784).float() / 255.0, targets


X_train, _ = to_matrix(train_full, train.indices)
X_val, _ = to_matrix(train_full, val.indices)
X_test, y_test = to_matrix(test)
print(f"train {tuple(X_train.shape)}  val {tuple(X_val.shape)}  test {tuple(X_test.shape)}")


def train_vae(latent_dim):
    """Build and train one VAE; return the model and its per-epoch history.

    Every decision during training looks at the validation set only.
    The test set is not touched here.
    """
    torch.manual_seed(42)                      # seed right before building the model
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    for epoch in range(1, EPOCHS + 1):
        tr_rec, tr_kl = run_epoch(model, X_train, BATCH_SIZE, optimizer, BETA)
        va_rec, va_kl = run_epoch(model, X_val, BATCH_SIZE)
        for k, v in zip(hist, (tr_rec, tr_kl, va_rec, va_kl)):
            hist[k].append(v)
        print(f"  d={latent_dim} epoch {epoch:2d}  train rec {tr_rec:7.2f} kl {tr_kl:6.2f}"
              f"  |  val rec {va_rec:7.2f} kl {va_kl:6.2f}")
    return model, hist


def test_numbers(model):
    """Final numbers on the test set: reconstruction, KL, negative ELBO."""
    rec, kl = run_epoch(model, X_test, BATCH_SIZE)
    return rec, kl, rec + BETA * kl


# =====================================================================
# Exercise 1 - B, C: the d = 16 VAE
# =====================================================================
print("=" * 60, "\nExercise 1 (d = 16)\n" + "=" * 60)
vae16, hist16 = train_vae(16)
rec16, kl16, nelbo16 = test_numbers(vae16)
print(f"TEST d=16: reconstruction {rec16:.2f}  KL {kl16:.2f}  negative ELBO {nelbo16:.2f} nats/image")
print(f"Validation KL: first epoch {hist16['val_kl'][0]:.2f}, last epoch {hist16['val_kl'][-1]:.2f}, "
      f"max {max(hist16['val_kl']):.2f} at epoch {int(np.argmax(hist16['val_kl'])) + 1}")

# Figure 1: two panels, reconstruction x epoch and KL x epoch, train and validation
epochs = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, term, label in [(axes[0], "rec", "Reconstruction (BCE)"), (axes[1], "kl", "KL")]:
    ax.plot(epochs, hist16[f"train_{term}"], marker="o", ms=3, label="Train")
    ax.plot(epochs, hist16[f"val_{term}"], marker="s", ms=3, label="Validation")
    ax.set(title=f"{label} per epoch (d = 16)", xlabel="Epoch", ylabel="nats per image")
    ax.set_xticks(epochs[::2])
    ax.legend()
fig.suptitle("Figure 1 - The two loss terms during training (d = 16)")
save(fig, "fig1.png")

# =====================================================================
# Exercise 1 - D: reconstructions and samples
# =====================================================================
vae16.eval()
# Figure 2: the first test image of each digit 0-9, and its reconstruction from mu
first_idx = [int((y_test == d).nonzero()[0]) for d in range(10)]
print("First test index of each digit 0-9:", first_idx)
with torch.no_grad():
    mu, _ = vae16.encode(X_test[first_idx])
    recon = vae16.decode(mu)                  # decode mu, not a sampled z
fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for d in range(10):
    axes[0, d].imshow(X_test[first_idx[d]].reshape(28, 28), cmap="gray")
    axes[1, d].imshow(recon[d].reshape(28, 28), cmap="gray")
    axes[0, d].set_title(f"digit {d}", fontsize=9)
    for r in range(2):
        axes[r, d].set_xticks([]); axes[r, d].set_yticks([])
axes[0, 0].set_ylabel("Original", fontsize=9)
axes[1, 0].set_ylabel("Reconstruction\n(decode mu)", fontsize=9)
fig.suptitle("Figure 2 - Test images (top) and their d = 16 reconstructions (bottom)")
save(fig, "fig2.png")

# Figure 3: 8 x 8 images decoded from z ~ N(0, I), after torch.manual_seed(0)
torch.manual_seed(0)
samples = vae16.sample(64)
fig, axes = plt.subplots(8, 8, figsize=(7, 7.4))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].reshape(28, 28), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Figure 3 - 64 samples decoded from z ~ N(0, I) (d = 16)")
save(fig, "fig3.png")

# Supporting numbers for the analysis (1E): how the encoded test set fills the prior
with torch.no_grad():
    mu_test16, logvar_test16 = vae16.encode(X_test)
    var_mu = mu_test16.var(dim=0)                          # spread of the codes per dimension
    mean_sigma2 = logvar_test16.exp().mean(dim=0)          # average posterior width
    kl_per_dim = 0.5 * (mu_test16.pow(2) + logvar_test16.exp() - 1 - logvar_test16).mean(dim=0)
    agg_var = var_mu + mean_sigma2                         # variance of the aggregate posterior
active = int((var_mu > 0.01).sum())
print(f"Active latent dimensions (Var[mu] > 0.01): {active} of 16")
print("KL per dimension (test, sorted):", np.round(np.sort(kl_per_dim.numpy())[::-1], 2))
print(f"Aggregate posterior variance per dim: min {agg_var.min():.3f}, max {agg_var.max():.3f}"
      f" (the prior has 1.000)")
act = var_mu > 0.01
sigma_active = logvar_test16[:, act].mul(0.5).exp().mean()
print(f"Mean posterior sigma on the active dimensions: {sigma_active:.3f} (prior sigma = 1)")
# Where do prior samples land? Distance from a prior sample to the nearest encoded
# test image, compared with the distance between neighbouring test images.
# A separate generator keeps the global seed sequence untouched.
z_prior = torch.randn(2000, 16, generator=torch.Generator().manual_seed(1))
d_prior = torch.cdist(z_prior, mu_test16).min(dim=1).values
d_test = torch.cdist(mu_test16[:2000], mu_test16)
d_test[torch.arange(2000), torch.arange(2000)] = float("inf")   # ignore the point itself
d_test = d_test.min(dim=1).values
print(f"Median distance to the nearest test code: from a prior sample {d_prior.median():.2f}, "
      f"from a test code {d_test.median():.2f} (ratio {d_prior.median() / d_test.median():.1f}x)")

# =====================================================================
# Exercise 2 - A: the d = 2 VAE, same model, data, seed, beta and budget
# =====================================================================
print("=" * 60, "\nExercise 2 (d = 2)\n" + "=" * 60)
vae2, hist2 = train_vae(2)
rec2, kl2, nelbo2 = test_numbers(vae2)
print(f"TEST d=2 : reconstruction {rec2:.2f}  KL {kl2:.2f}  negative ELBO {nelbo2:.2f} nats/image")
print(f"TEST d=16: reconstruction {rec16:.2f}  KL {kl16:.2f}  negative ELBO {nelbo16:.2f} nats/image")

# =====================================================================
# Exercise 2 - B: Figure 4, mu of all 10 000 test images
# =====================================================================
vae2.eval()
with torch.no_grad():
    mu2, _ = vae2.encode(X_test)
mu2_np, y_np = mu2.numpy(), y_test.numpy()
fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for d in range(10):
    pts = mu2_np[y_np == d]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, color=cmap(d), label=str(d))
ax.set(title="Figure 4 - Encoded test set: mu of the 10 000 test images (d = 2)",
       xlabel="$\\mu_1$", ylabel="$\\mu_2$")
ax.legend(title="Digit", markerscale=4, loc="upper right", fontsize=8)
save(fig, "fig4.png")

# Supporting numbers (2E): which digits overlap? Nearest-centroid confusion in the plane
centroids = np.stack([mu2_np[y_np == d].mean(0) for d in range(10)])
nearest = np.argmin(((mu2_np[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
conf = np.zeros((10, 10), dtype=int)
np.add.at(conf, (y_np, nearest), 1)
pairs = [(conf[i, j] + conf[j, i], i, j) for i in range(10) for j in range(i + 1, 10)]
pairs.sort(reverse=True)
print("Centroids (mu1, mu2):")
for d in range(10):
    print(f"  {d}: ({centroids[d, 0]:+.2f}, {centroids[d, 1]:+.2f})")
print(f"Nearest-centroid accuracy in the 2-D plane: {(nearest == y_np).mean():.4f}")
print("Most-overlapping digit pairs (test images assigned to the other's centroid):")
for n, i, j in pairs[:5]:
    print(f"  {i} <-> {j}: {n}")

# =====================================================================
# Exercise 2 - C: Figure 5, the decoded plane on a Phi^-1 grid
# =====================================================================
q = np.linspace(0.05, 0.95, 15)
z_vals = norm.ppf(q) + 0.0            # equal prior mass between neighbouring grid lines (+0.0 avoids "-0.00")
print("Grid values z = Phi^-1(q):", np.round(z_vals, 3))
canvas = np.zeros((15 * 28, 15 * 28))
with torch.no_grad():
    for row, z2 in enumerate(z_vals[::-1]):          # top row = largest z2, as in Figure 4
        zs = torch.tensor([[z1, z2] for z1 in z_vals], dtype=torch.float32)
        imgs = vae2.decode(zs).reshape(15, 28, 28).numpy()
        for col in range(15):
            canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = imgs[col]
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.imshow(canvas, cmap="gray")
ticks = np.arange(15) * 28 + 14
ax.set_xticks(ticks, [f"{v:.2f}".replace("-0.00", "0.00") for v in z_vals], rotation=90, fontsize=7)
ax.set_yticks(ticks, [f"{v:.2f}".replace("-0.00", "0.00") for v in z_vals[::-1]], fontsize=7)
ax.set(title="Figure 5 - Decoded plane, 15 x 15 grid with z = $\\Phi^{-1}(q)$ (d = 2)",
       xlabel="$z_1$", ylabel="$z_2$")
save(fig, "fig5.png")

# =====================================================================
# Exercise 2 - D: Figure 6, interpolation in pixel space vs latent space (d = 16)
# =====================================================================
IDX_A, IDX_B = 3, 2                    # test images of two different digits
print(f"Interpolated test indices: a = {IDX_A} (digit {int(y_test[IDX_A])}), "
      f"b = {IDX_B} (digit {int(y_test[IDX_B])})")
ts = np.linspace(0, 1, 10)
x_a, x_b = X_test[IDX_A], X_test[IDX_B]
with torch.no_grad():
    mu_ab, _ = vae16.encode(torch.stack([x_a, x_b]))
    pixel_row = [(1 - t) * x_a + t * x_b for t in ts]
    latent_row = vae16.decode(torch.stack([(1 - t) * mu_ab[0] + t * mu_ab[1] for t in ts]))
fig, axes = plt.subplots(2, 10, figsize=(12, 3.0))
for i, t in enumerate(ts):
    axes[0, i].imshow(pixel_row[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(latent_row[i].reshape(28, 28), cmap="gray")
    axes[0, i].set_title(f"t = {t:.2f}", fontsize=8)
    for r in range(2):
        axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
axes[0, 0].set_ylabel("Pixel space", fontsize=9)
axes[1, 0].set_ylabel("Latent space\n(d = 16)", fontsize=9)
fig.suptitle(f"Figure 6 - Interpolating test image {IDX_A} (digit {int(y_test[IDX_A])}) "
             f"into test image {IDX_B} (digit {int(y_test[IDX_B])})")
save(fig, "fig6.png")

print("\nFigures written to", FIG_DIR)
