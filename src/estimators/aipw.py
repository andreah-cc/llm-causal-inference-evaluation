"""Augmented inverse-probability weighting (doubly robust) baseline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from .propensity import estimate_propensity_logit, overlap_status

def baseline_aipw_dr(
    df: pd.DataFrame,
    clip: float = 1e-3,
    outcome_model: str = "linear",   # "linear" or "ridge"
    ridge_alpha: float = 1.0
):
    """
    Doubly Robust AIPW estimator for ATE.

    Uses:
      e_hat(x) = P(A=1|X)  (from estimate_propensity_logit)
      mu1_hat(x) = E[Y|A=1,X]
      mu0_hat(x) = E[Y|A=0,X]

    AIPW score:
      psi_i = (mu1_hat - mu0_hat)
              + A*(Y - mu1_hat)/e_hat
              - (1-A)*(Y - mu0_hat)/(1-e_hat)

    ate_hat = mean(psi_i)
    se_hat  = sqrt(var(psi_i)/n)
    """

    # --- reuse your propensity code (also drops missing rows) ---
    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]

    # overlap diagnostics
    status, e_min, e_max = overlap_status(e_hat)

    # choose outcome regressor
    if outcome_model == "ridge":
        reg = lambda: Ridge(alpha=ridge_alpha)
    else:
        reg = lambda: LinearRegression()

    # --- estimate mu0_hat, mu1_hat ---
    if len(x_cols) == 0:
        # no covariates: fallback to group means (still valid as a baseline)
        mu1 = float(np.mean(Y[A == 1])) if np.any(A == 1) else None
        mu0 = float(np.mean(Y[A == 0])) if np.any(A == 0) else None
        if mu1 is None or mu0 is None:
            return {
                "estimator_name": "aipw_dr",
                "ate_hat": None,
                "se_hat": None,
                "ci_95": [None, None],
                "n_used": int(n),
                "p_covariates": 0,
                "overlap_status": "violation",
                "e_min": e_min,
                "e_max": e_max,
                "notes": "No treated or no control units after dropping missing rows."
            }
        mu1_hat = np.full(n, mu1, dtype=float)
        mu0_hat = np.full(n, mu0, dtype=float)

    else:
        X = d[x_cols].to_numpy(dtype=float)

        idx1 = (A == 1)
        idx0 = (A == 0)

        if not np.any(idx1) or not np.any(idx0):
            return {
                "estimator_name": "aipw_dr",
                "ate_hat": None,
                "se_hat": None,
                "ci_95": [None, None],
                "n_used": int(n),
                "p_covariates": int(len(x_cols)),
                "overlap_status": "violation",
                "e_min": e_min,
                "e_max": e_max,
                "notes": "No treated or no control units after dropping missing rows."
            }

        # Fit separate outcome models
        m1 = reg()
        m0 = reg()
        m1.fit(X[idx1], Y[idx1])
        m0.fit(X[idx0], Y[idx0])

        mu1_hat = m1.predict(X).astype(float)
        mu0_hat = m0.predict(X).astype(float)

    # --- AIPW / DR score and estimate ---
    # (e_hat is already clipped in estimate_propensity_logit)
    psi = (mu1_hat - mu0_hat) + (A * (Y - mu1_hat) / e_hat) - ((1 - A) * (Y - mu0_hat) / (1 - e_hat))

    ate_hat = float(np.mean(psi))
    se_hat = float(np.sqrt(np.var(psi, ddof=1) / n)) if n > 1 else None
    ci_95 = [ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat] if se_hat is not None else [None, None]

    notes = []
    notes.append(f"Outcome model: {outcome_model}")
    if status != "ok":
        notes.append(f"Overlap {status}: e_hat in [{e_min:.4g}, {e_max:.4g}] (weights may be unstable).")

    return {
        "estimator_name": "aipw_dr",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": [float(ci_95[0]), float(ci_95[1])] if se_hat is not None else ci_95,
        "n_used": int(n),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": " ".join(notes) if notes else None
    }
