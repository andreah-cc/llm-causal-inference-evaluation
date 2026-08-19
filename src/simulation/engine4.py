"""Engine 4: High-dimensional sparse confounding with heterogeneous treatment effects."""

import numpy as np
from .common import sigmoid


def _normalize_index_set(index_set, p):
    values = np.asarray(index_set) if index_set is not None else np.array([], dtype=int)
    if values.size == p and values.ndim == 1:
        values = np.where(values.astype(bool))[0]
    else:
        values = values.astype(int).ravel()
    return values[values < min(5, p)]


def gen_treat_4(X, SA, w, a, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p, _ = X.shape
    SA = _normalize_index_set(SA, p)
    w = np.asarray(w, float).reshape(-1, 1)

    score = a if SA.size == 0 else a + k * (w[SA].T @ X[SA])
    e = sigmoid(score)
    return rng.binomial(1, e).astype(int)


def gen_outcome_4(X, A, SY, b, t0, t1, j_star, eps):
    p, n = X.shape
    SY = _normalize_index_set(SY, p)
    b = np.asarray(b, float).reshape(-1, 1)

    m0 = np.zeros((1, n)) if SY.size == 0 else (b[SY].T @ X[SY])
    tau_x = t0 + t1 * X[int(j_star):int(j_star) + 1, :]

    Y0 = m0 + eps
    Y1 = m0 + tau_x + eps
    Y = m0 + A * tau_x + eps
    return Y0, Y1, Y


def main_4(X, SA, SY, w, a, k, b, t0, t1, j_star, eps, rng=None):
    A = gen_treat_4(X, SA, w, a, k, rng=rng)
    Y0, Y1, Y = gen_outcome_4(X, A, SY, b, t0, t1, j_star, eps)
    return X, A, Y0, Y1, Y
