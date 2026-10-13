"""Exercise 1: numerical check of the hand-derived backpropagation step.

This script only VERIFIES the numbers derived by hand in the report.
It does not use any library model, just numpy arithmetic.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# Given values
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# A - forward pass
z1 = W1 @ x + b1            # hidden pre-activations
h1 = np.tanh(z1)            # hidden activations
u2 = W2 @ h1 + b2           # output pre-activation
y_hat = np.tanh(u2)         # output
L = (y - y_hat) ** 2        # MSE with N = 1
print("z1    =", z1)
print("h1    =", h1)
print(f"u2    = {u2:.6f}")
print(f"y_hat = {y_hat:.6f}")
print(f"L     = {L:.6f}")

# B - backward pass
dL_dyhat = -2 * (y - y_hat)                 # derivative of (y - y_hat)^2
dL_du2 = dL_dyhat * (1 - y_hat ** 2)        # tanh'(u) = 1 - tanh(u)^2
dL_dW2 = dL_du2 * h1                        # u2 = W2 . h1 + b2
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2                        # back through W2
dL_dz1 = dL_dh1 * (1 - h1 ** 2)             # back through hidden tanh
dL_dW1 = np.outer(dL_dz1, x)                # z1 = W1 x + b1
dL_db1 = dL_dz1
print(f"\ndL/dy_hat = {dL_dyhat:.6f}")
print(f"dL/du2    = {dL_du2:.6f}")
print("dL/dW2    =", dL_dW2)
print(f"dL/db2    = {dL_db2:.6f}")
print("dL/dh1    =", dL_dh1)
print("dL/dz1    =", dL_dz1)
print("dL/dW1    =\n", dL_dW1)
print("dL/db1    =", dL_db1)

# C - parameter update (gradient descent)
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
print("\nW2_new =", W2_new)
print(f"b2_new = {b2_new:.6f}")
print("W1_new =\n", W1_new)
print("b1_new =", b1_new)

# Second forward pass with the updated parameters
h1_new = np.tanh(W1_new @ x + b1_new)
y_hat_new = np.tanh(W2_new @ h1_new + b2_new)
L_new = (y - y_hat_new) ** 2
print(f"\nSecond pass: y_hat = {y_hat_new:.6f}, L = {L_new:.6f} (before: {L:.6f})")
