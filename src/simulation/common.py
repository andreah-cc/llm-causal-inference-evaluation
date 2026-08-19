"""Shared utilities for the simulation benchmark."""

from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    """Numerically simple logistic/sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-x))


def gen_covariate(p, n, sigma=None, rng=None):
    """Generate p x n Gaussian covariates."""
    if rng is None:
        rng = np.random.default_rng()
    if sigma is None:
        sigma = np.eye(p)
    return rng.multivariate_normal(
        mean=np.zeros(p),
        cov=sigma,
        size=n,
    ).T


def gen_error(n, sd=1.0, rng=None):
    """Generate 1 x n Gaussian outcome noise."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=sd, size=(1, n))


def save_data(D, filepath):
    """
    Save a standard simulation tuple (X, A, Y0, Y1, Y) to CSV.

    X is expected to be p x n; the remaining arrays may be 1 x n or length n.
    """
    X, A, Y0, Y1, Y = D
    X = np.asarray(X).T
    A = np.asarray(A).reshape(-1, 1)
    Y0 = np.asarray(Y0).reshape(-1, 1)
    Y1 = np.asarray(Y1).reshape(-1, 1)
    Y = np.asarray(Y).reshape(-1, 1)

    full_data = np.hstack([X, A, Y0, Y1, Y])
    columns = [f"X{i+1}" for i in range(X.shape[1])] + ["A", "Y_0", "Y_1", "Y"]

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(full_data, columns=columns).to_csv(filepath, index=False)
    return filepath
