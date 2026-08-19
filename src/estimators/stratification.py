"""Propensity-score stratification baseline."""

import numpy as np
import pandas as pd

from .propensity import estimate_propensity_logit, overlap_status

def baseline_ps_stratification(df: pd.DataFrame, K: int = 5, clip: float = 1e-3):
    """
    Stratify by quantiles of e_hat into K bins.
    """
    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]

    qs = np.linspace(0, 1, K + 1)
    edges = np.unique(np.quantile(e_hat, qs))

    if len(edges) < 3:
        bins = np.zeros(n, dtype=int)
        K_eff = 1
    else:
        bins = pd.cut(e_hat, bins=edges, include_lowest=True, labels=False).to_numpy()
        K_eff = int(np.nanmax(bins) + 1)

    ate = 0.0
    var = 0.0

    for b in range(K_eff):
        idx = (bins == b)
        nb = int(np.sum(idx))
        if nb == 0:
            continue

        w_k = nb / n
        y1 = Y[idx & (A == 1)]
        y0 = Y[idx & (A == 0)]
        n1, n0 = len(y1), len(y0)

        if n1 == 0 or n0 == 0:
            status, e_min, e_max = overlap_status(e_hat)
            return {
                "estimator_name": "ps_strat",
                "ate_hat": None,
                "se_hat": None,
                "ci_95": [None, None],
                "n_used": int(n),
                "p_covariates": int(len(x_cols)),
                "overlap_status": "violation",
                "e_min": e_min,
                "e_max": e_max,
                "notes": f"Positivity violation: bin {b} has n_treated={n1}, n_control={n0} (K_eff={K_eff})."
            }

        diff_k = float(np.mean(y1) - np.mean(y0))
        ate += w_k * diff_k

        s1 = float(np.var(y1, ddof=1)) if n1 > 1 else 0.0
        s0 = float(np.var(y0, ddof=1)) if n0 > 1 else 0.0
        var += (w_k ** 2) * (s1 / n1 + s0 / n0)

    ate_hat = float(ate)
    se_hat = float(np.sqrt(var)) if np.isfinite(var) else None
    ci_95 = [ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat] if se_hat is not None else [None, None]

    status, e_min, e_max = overlap_status(e_hat)

    return {
        "estimator_name": "ps_strat",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": [float(ci_95[0]), float(ci_95[1])] if se_hat is not None else ci_95,
        "n_used": int(n),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": f"K_eff={K_eff}"
    }
