"""Engine 8: Measurement error plus heavy-tailed outcome noise."""

import numpy as np
from .common import sigmoid


def mixed_normal_noise(n, sd, w_main=0.95, sd_multiplier=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    main = rng.random(n) < w_main
    eps = rng.normal(0.0, sd, size=n)
    tail_n = (~main).sum()
    if tail_n > 0:
        eps[~main] = rng.normal(0.0, sd_multiplier * sd, size=tail_n)
    return eps.reshape(1, n)


def main_8(
    X, w, a, k, b, tau, sd,
    sigma_u=0.5, w_main=0.95, sd_multiplier=10.0, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    p, n = X.shape
    w = np.asarray(w, float).reshape(p, 1)
    b = np.asarray(b, float).reshape(p, 1)
    w = w / np.linalg.norm(w)
    b = b / np.linalg.norm(b)

    score = a + k * (w.T @ X)
    e = sigmoid(score)
    A = rng.binomial(1, e).astype(int)

    U = rng.normal(0.0, sigma_u, size=(p, n))
    W = X + U

    eps = mixed_normal_noise(
        n, sd, w_main=w_main, sd_multiplier=sd_multiplier, rng=rng
    )

    mu = b.T @ W
    Y0 = mu + eps
    Y1 = mu + tau + eps
    Y = Y0 * (1 - A) + Y1 * A
    return W, A, Y0, Y1, Y
