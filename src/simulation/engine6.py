"""Engine 6: Nonlinear treatment assignment and nonlinear baseline outcome."""

import numpy as np
from .common import sigmoid


def gen_covar_6(p, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1, size=(n, p))


def gen_treat_6(X, a0, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if X.shape[1] < 5:
        raise ValueError("Engine 6 requires p >= 5 because it uses X1..X5.")

    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    score = x1 + x2**2 - x3 * x4 + 0.5 * np.sin(x5)
    e = sigmoid(a0 + k * score)
    A = rng.binomial(1, e, size=X.shape[0])
    return A, e


def gen_outcome_6(X, A, tau0, sd, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if X.shape[1] < 4:
        raise ValueError("Engine 6 outcome requires p >= 4.")

    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    m0 = 0.5 * x1 + 0.25 * x2**2 - 0.2 * x3 * x4
    eps = rng.normal(0.0, sd, size=X.shape[0])
    Y0 = m0 + eps
    Y1 = m0 + tau0 + eps
    Y = Y0 * (1 - A) + Y1 * A
    return Y0, Y1, Y


def main_6(p, n, a0, k, tau0, sd, seed=0, return_propensity=True):
    rng = np.random.default_rng(seed)
    X = gen_covar_6(p, n, rng=rng)
    A, e = gen_treat_6(X, a0, k, rng=rng)
    Y0, Y1, Y = gen_outcome_6(X, A, tau0, sd, rng=rng)
    D = (X.T, A, Y0, Y1, Y)
    if return_propensity:
        return D, e
    return D
