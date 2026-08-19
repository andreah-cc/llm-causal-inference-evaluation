"""Nearest-neighbor propensity-score matching baseline."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .propensity import estimate_propensity_logit, overlap_status

def baseline_ps_matching(df: pd.DataFrame, clip: float = 1e-3, bootstrap_B: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)

    treated = np.where(A == 1)[0]
    control = np.where(A == 0)[0]
    if len(treated) == 0 or len(control) == 0:
        status, e_min, e_max = overlap_status(e_hat)
        return {
            "estimator_name": "ps_match_att",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "n_used": int(d.shape[0]),
            "p_covariates": int(len(x_cols)),
            "overlap_status": "violation",
            "e_min": e_min,
            "e_max": e_max,
            "notes": "No treated or no control units."
        }

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(e_hat[control].reshape(-1, 1))
    _, idx = nn.kneighbors(e_hat[treated].reshape(-1, 1))
    matched_controls = control[idx[:, 0]]

    diffs = Y[treated] - Y[matched_controls]
    ate_hat = float(np.mean(diffs))

    se_hat = None
    ci_95 = [None, None]
    if bootstrap_B and bootstrap_B > 0:
        boots = []
        n_t = len(diffs)
        for _ in range(int(bootstrap_B)):
            samp = rng.integers(0, n_t, size=n_t)
            boots.append(float(np.mean(diffs[samp])))
        se_hat = float(np.std(boots, ddof=1))
        ci_95 = [ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat]

    status, e_min, e_max = overlap_status(e_hat)

    return {
        "estimator_name": "ps_match_att",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": [float(ci_95[0]), float(ci_95[1])] if se_hat is not None else ci_95,
        "n_used": int(d.shape[0]),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": "ATT-style (treated matched to controls)."
    }
