"""A variational autoencoder for MNIST, in PyTorch.

Autograd is allowed in this activity, so the layers and the optimizer come
from torch. The VAE-specific parts are written by hand, as required:

- the reparameterization  z = mu + sigma * eps   (reparameterize)
- the reconstruction term, BCE summed over the 784 pixels   (reconstruction_loss)
- the closed-form KL between N(mu, sigma^2) and N(0, I)    (kl_divergence)
- sampling from the prior                                   (sample)

Every loss value returned here is in nats per image: summed over pixels
(or latent dimensions) and averaged over the images of the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 784 -> 512 -> 256, ReLU after each
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # Two heads 256 -> d: one for mu, one for log sigma^2
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        # Decoder: d -> 256 -> 512 -> 784, ReLU on hidden layers, sigmoid at the output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        # Output = 784 pixel probabilities (Bernoulli means), each in (0, 1)
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + sigma * eps, eps ~ N(0, I).
        # The randomness lives in eps, which does not depend on the parameters,
        # so gradients flow through mu and sigma as ordinary arithmetic.
        sigma = torch.exp(0.5 * logvar)      # sigma = sqrt(exp(log sigma^2))
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n):
        # Draw z from the prior N(0, I) and decode it
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


def reconstruction_loss(x_hat, x):
    """BCE summed over the 784 pixels, averaged over the batch (nats/image).

    reduction="sum" and then / batch size. reduction="mean" would also average
    over pixels and shrink this term 784x relative to the KL.
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]


def kl_divergence(mu, logvar):
    """Closed-form KL( N(mu, sigma^2) || N(0, I) ), summed over the latent
    dimensions and averaged over the batch (nats/image):

        1/2 * sum_i ( mu_i^2 + sigma_i^2 - 1 - log sigma_i^2 )
    """
    kl_per_image = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)
    return kl_per_image.mean()


def run_epoch(model, X, batch_size, optimizer=None, beta=1.0):
    """One pass over X. Trains if an optimizer is given, otherwise only evaluates.

    Returns the epoch averages (reconstruction, KL) in nats per image,
    weighted by batch size so a smaller last batch counts correctly.
    """
    training = optimizer is not None
    model.train(training)
    n = X.shape[0]
    # Shuffle only when training; uses torch's global RNG (seeded by the caller)
    order = torch.randperm(n) if training else torch.arange(n)
    rec_sum, kl_sum = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for start in range(0, n, batch_size):
            xb = X[order[start:start + batch_size]]
            x_hat, mu, logvar = model(xb)
            rec = reconstruction_loss(x_hat, xb)
            kl = kl_divergence(mu, logvar)
            loss = rec + beta * kl          # the negative ELBO for this batch
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            rec_sum += rec.item() * xb.shape[0]
            kl_sum += kl.item() * xb.shape[0]
    return rec_sum / n, kl_sum / n
