"""Engine 2: Nonlinear outcome with a constant treatment effect."""

import numpy as np
from .common import sigmoid


def gen_treat_2(X, w, a, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)
    w = w.reshape(-1, 1)
    score = a + k * (w.T @ X)
    e = sigmoid(score)
    return rng.binomial(1, e)


def gen_outcome_2(X, A, tau, eps):
    if X.shape[0] < 5:
        raise ValueError("Engine 2 requires p >= 5 because it uses X1..X5.")

    x1, x2, x3, x4, x5 = X[0], X[1], X[2], X[3], X[4]
    g = 0.5 * x1**2 + np.sin(x2) + 0.25 * x3 * x4 - 0.1 * x5**3

    Y0 = g + eps
    Y1 = g + tau + eps
    Y = g + tau * A + eps
    return Y0, Y1, Y


def main_2(X, w, a, k, tau, eps, rng=None):
    A = gen_treat_2(X, w, a, k, rng=rng)
    Y0, Y1, Y = gen_outcome_2(X, A, tau, eps)
    return X, A, Y0, Y1, Y
