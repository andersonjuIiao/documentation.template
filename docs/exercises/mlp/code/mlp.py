"""A multi-layer perceptron written from scratch with numpy.

The same class is used, untouched, in Exercises 2, 3 and 4. Only the
arguments change: the list of layer sizes, the output type ("sigmoid"
for binary, "softmax" for multi-class), the learning rate and the rng.

Conventions
-----------
X : (N, d_in)  one sample per row
Y : (N, K)     targets; K = 1 for binary (0/1), one-hot for multi-class
W[l] : (n_in, n_out), b[l] : (n_out,)  so that Z = A @ W + b
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes, output="sigmoid", rng=None):
        # layer_sizes, e.g. [2, 8, 1] or [4, 16, 16, 3]
        self.layer_sizes = list(layer_sizes)
        self.output = output
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Xavier/Glorot init: keeps tanh pre-activations near the linear zone
            std = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(self.rng.normal(0.0, std, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ---------- activations ----------
    @staticmethod
    def _sigmoid(u):
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def _softmax(u):
        u = u - u.max(axis=1, keepdims=True)  # numerical stability, same result
        e = np.exp(u)
        return e / e.sum(axis=1, keepdims=True)

    def _output_activation(self, u):
        if self.output == "sigmoid":
            return self._sigmoid(u)
        return self._softmax(u)

    # ---------- forward ----------
    def forward(self, X):
        """Return the output probabilities and cache every layer's activation."""
        A = X
        self.cache = [A]  # A^(0) = X, then h^(1), h^(2), ...
        n_layers = len(self.W)
        for l in range(n_layers):
            Z = A @ self.W[l] + self.b[l]
            if l < n_layers - 1:
                A = np.tanh(Z)                     # hidden layers: tanh
            else:
                A = self._output_activation(Z)     # output layer: sigmoid/softmax
            self.cache.append(A)
        return A

    # ---------- loss ----------
    def loss(self, P, Y):
        """Binary cross-entropy (sigmoid) or categorical cross-entropy (softmax)."""
        eps = 1e-12
        P = np.clip(P, eps, 1 - eps)
        N = Y.shape[0]
        if self.output == "sigmoid":
            return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / N
        return -np.sum(Y * np.log(P)) / N

    # ---------- backward ----------
    def backward(self, Y):
        """Gradients of the loss w.r.t. every W and b, by the chain rule.

        For BOTH pairs (sigmoid + BCE) and (softmax + CE) the gradient of the
        loss w.r.t. the output pre-activation is the same:  dL/dU = (P - Y) / N.
        That is why this method does not need to know which output is used.
        """
        N = Y.shape[0]
        P = self.cache[-1]
        delta = (P - Y) / N                       # dL/dU at the output layer
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            A_prev = self.cache[l]                # input to layer l
            grads_W[l] = A_prev.T @ delta         # dL/dW = A_prev^T delta
            grads_b[l] = delta.sum(axis=0)        # dL/db = sum over samples
            if l > 0:
                dA_prev = delta @ self.W[l].T     # dL/dh, back through W
                delta = dA_prev * (1 - A_prev ** 2)  # times tanh'(z) = 1 - h^2
        return grads_W, grads_b

    # ---------- update ----------
    def update(self, grads_W, grads_b, lr):
        """Plain gradient descent: theta <- theta - lr * dL/dtheta."""
        for l in range(len(self.W)):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    # ---------- training loop ----------
    def fit(self, X, Y, epochs, lr):
        """Full-batch gradient descent; returns the training loss per epoch."""
        history = []
        for _ in range(epochs):
            P = self.forward(X)
            history.append(self.loss(P, Y))
            grads_W, grads_b = self.backward(Y)
            self.update(grads_W, grads_b, lr)
        return np.array(history)

    # ---------- prediction ----------
    def predict(self, X):
        P = self.forward(X)
        if self.output == "sigmoid":
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))
