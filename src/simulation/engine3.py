"""Engine 3: Interaction-heavy heterogeneous treatment effects."""

import numpy as np
from .common import sigmoid


def gen_covar_3(p, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0.0, 1.0, size=(n, p))


def gen_treat_3(X, v, alpha, kappa, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    score = alpha + kappa * (X @ v)
    e = sigmoid(score)
    return rng.binomial(1, e, size=X.shape[0]).astype(int)


def m0_engine3(X):
    if X.shape[1] < 4:
        raise ValueError("Engine 3 baseline requires p >= 4.")
    return 0.2 * X[:, 0] + 0.2 * X[:, 1] + 0.2 * (X[:, 2] * X[:, 3])


def tau_engine3(X, tau0, tau1, tau2):
    if X.shape[1] < 3:
        raise ValueError("Engine 3 treatment effect requires p >= 3.")
    return tau0 + tau1 * X[:, 0] + tau2 * (X[:, 2] ** 2 - 1.0)


def gen_outcome_3(X, A, tau0, tau1, tau2, sd, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    eps = rng.normal(0.0, sd, size=X.shape[0])
    base = m0_engine3(X)
    tau_x = tau_engine3(X, tau0, tau1, tau2)
    Y0 = base + eps
    Y1 = base + tau_x + eps
    Y = Y0 * (1 - A) + Y1 * A
    return Y0, Y1, Y


def main_3(p, n, v, alpha, kappa, tau0, tau1, tau2, sd, seed=0):
    rng = np.random.default_rng(seed)
    X = gen_covar_3(p, n, rng=rng)
    A = gen_treat_3(X, v, alpha, kappa, rng=rng)
    Y0, Y1, Y = gen_outcome_3(X, A, tau0, tau1, tau2, sd, rng=rng)
    return X.T, A, Y0, Y1, Y
