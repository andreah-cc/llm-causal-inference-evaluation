"""Inverse-probability weighting baseline."""

import numpy as np
import pandas as pd

from .propensity import estimate_propensity_logit, overlap_status

def baseline_ipw(df: pd.DataFrame, clip: float = 1e-3):
    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]

    w1 = A / e_hat
    w0 = (1 - A) / (1 - e_hat)

    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    ate_hat = float(mu1 - mu0)

    # baseline-grade SE
    mw1, mw0 = np.mean(w1), np.mean(w0)
    psi = (w1 / mw1) * (Y - mu1) - (w0 / mw0) * (Y - mu0)
    se_hat = float(np.sqrt(np.var(psi, ddof=1) / n)) if n > 1 else None
    ci_95 = [ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat] if se_hat is not None else [None, None]

    status, e_min, e_max = overlap_status(e_hat)

    return {
        "estimator_name": "ipw",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": [float(ci_95[0]), float(ci_95[1])] if se_hat is not None else ci_95,
        "n_used": int(n),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": None
    }
