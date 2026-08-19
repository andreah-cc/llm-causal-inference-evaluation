"""Engine 1: Linear baseline outcome with tunable confounding."""

import numpy as np
from .common import sigmoid


def gen_treat_1(X, w, a, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)
    w = w.reshape(-1, 1)
    score = a + k * (w.T @ X)
    e = sigmoid(score)
    return rng.binomial(1, e)


def gen_outcome_1(X, A, b, tau, eps):
    b = np.asarray(b, float)
    b = b / np.linalg.norm(b)
    b = b.reshape(-1, 1)
    baseline = b.T @ X
    Y0 = baseline + eps
    Y1 = baseline + tau + eps
    Y = baseline + tau * A + eps
    return Y0, Y1, Y


def main_1(X, w, a, k, b, tau, eps, rng=None):
    A = gen_treat_1(X, w, a, k, rng=rng)
    Y0, Y1, Y = gen_outcome_1(X, A, b, tau, eps)
    return X, A, Y0, Y1, Y
