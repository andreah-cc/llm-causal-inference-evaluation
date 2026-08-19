"""Bootstrap uncertainty helpers used by Module 3 estimators."""

import numpy as np
import pandas as pd

def _bootstrap_ate(d, ate_fn, B=300, seed=123, ci_level=0.95):
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    draws = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        d_b = d.iloc[idx].copy()
        draws[b] = float(ate_fn(d_b))

    se = float(np.std(draws, ddof=1))

    alpha = 1 - ci_level
    ci_low = float(np.quantile(draws, alpha / 2))
    ci_high = float(np.quantile(draws, 1 - alpha / 2))

    return se, [ci_low, ci_high]

def _bootstrap_estimates(d: pd.DataFrame, estimator_fn, B: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(d)
    vals = []

    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        d_b = d.iloc[idx].reset_index(drop=True)
        try:
            val = estimator_fn(d_b)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
        except Exception:
            continue

    return np.asarray(vals, dtype=float)

def _bootstrap_ci(ate_hat: float, boot: np.ndarray, ci_method: str = "percentile"):
    if len(boot) < 2:
        return None, [None, None], "Too few valid bootstrap replicates."

    se_hat = float(np.std(boot, ddof=1))

    if ci_method == "normal":
        ci = [float(ate_hat - 1.96 * se_hat), float(ate_hat + 1.96 * se_hat)]
    else:
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    return se_hat, ci, f"Whole-procedure bootstrap used with {len(boot)} valid replicates."
