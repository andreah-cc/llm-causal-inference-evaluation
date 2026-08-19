"""Engine 7: Missing-outcome stress tests under MCAR and MAR."""

import numpy as np
from .common import sigmoid


def engine7_mcar(X, A, Y, pi, seed=None):
    """Observe Y with probability pi independently of X, A, and Y."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    R = rng.binomial(1, pi, size=n).astype(int)
    Y_obs = np.asarray(Y, float).copy()
    Y_obs[R == 0] = np.nan
    return X, A, Y_obs, R


def engine7_mar(X, A, Y, gamma0, gammaA, gamma, seed=None):
    """Observe Y under a MAR mechanism depending on X and A."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    A = np.asarray(A)
    gamma = np.asarray(gamma)

    scores = gamma0 + gammaA * A + X @ gamma
    p_obs = sigmoid(scores)
    R = rng.binomial(1, p_obs, size=len(Y)).astype(int)

    Y_obs = np.asarray(Y, float).copy()
    Y_obs[R == 0] = np.nan
    return X, A, Y_obs, R
