"""Single-layer perceptron, implemented from scratch (NumPy only).

Written once here (Exercise 1, item B) and reused unchanged in
Exercise 2, with the pocket algorithm switched on via
`fit(..., track_pocket=True)`.
"""

import numpy as np


class Perceptron:
    """Binary perceptron with {0, 1} labels.

    Prediction:  y_hat = step(w . x + b), step(z) = 1 if z >= 0 else 0
    Update rule: w <- w + eta * (y - y_hat) * x
                 b <- b + eta * (y - y_hat)
    """

    def __init__(self, n_features, rng, eta=0.01):
        self.eta = eta

        # Non-zero init: with w = 0 the learning rate only rescales w,
        # so it would never change the decision boundary or epoch count
        # (see Exercise 1, item D3).
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

        # Accuracy on the full dataset, recorded after every epoch,
        # for the weights as they stand at that point in training.
        self.history_acc = []

        # Pocket algorithm bookkeeping (only populated when
        # fit(..., track_pocket=True) is used, i.e. Exercise 2).
        self.pocket_w = self.w.copy()
        self.pocket_b = self.b
        self.pocket_acc = -1.0
        self.pocket_epoch = 0
        self.pocket_history_acc = []

    def _predict_one(self, x):
        z = x @ self.w + self.b
        return 1 if z >= 0 else 0

    def predict(self, X, w=None, b=None):
        w = self.w if w is None else w
        b = self.b if b is None else b
        z = X @ w + b
        return np.where(z >= 0, 1, 0)

    def accuracy(self, X, y, w=None, b=None):
        return float(np.mean(self.predict(X, w, b) == y))

    def fit(self, X, y, max_epochs=100, track_pocket=False):
        """Train for up to `max_epochs`, stopping early on a clean pass.

        Samples are visited in dataset order (no reshuffling), so the
        run is fully determined by the seed used to build X, y and w.
        """
        n_samples = X.shape[0]

        for epoch in range(1, max_epochs + 1):
            updated_this_epoch = False

            for i in range(n_samples):
                xi, yi = X[i], y[i]
                y_hat = self._predict_one(xi)
                error = yi - y_hat  # 0, +1 or -1

                if error != 0:
                    self.w = self.w + self.eta * error * xi
                    self.b = self.b + self.eta * error
                    updated_this_epoch = True

                    if track_pocket:
                        acc = self.accuracy(X, y)
                        if acc > self.pocket_acc:
                            self.pocket_acc = acc
                            self.pocket_w = self.w.copy()
                            self.pocket_b = self.b
                            self.pocket_epoch = epoch

            epoch_acc = self.accuracy(X, y)
            self.history_acc.append(epoch_acc)
            if track_pocket:
                self.pocket_history_acc.append(self.pocket_acc)

            if not updated_this_epoch:
                break

        return self
