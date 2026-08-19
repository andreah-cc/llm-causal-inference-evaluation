"""Doubly robust AIPW estimator with several outcome-model choices."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from .compat import estimate_propensity_logit

def baseline_aipw_dr(
    df: pd.DataFrame,
    clip: float = 1e-3,
    outcome_model: str = "linear",   # "linear", "ridge", "rf", "hgb"
    ridge_alpha: float = 1.0,
    rf_n_estimators: int = 300,
    rf_min_samples_leaf: int = 5,
    rf_max_depth=None,
    hgb_max_depth: int = 3,
    hgb_learning_rate: float = 0.05,
    hgb_max_iter: int = 400,
    random_state: int = 0
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

    d, x_cols, e_hat, ps_info = estimate_propensity_logit(df, clip=clip)
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]
    # overlap diagnostics (already computed inside estimate_propensity_logit)
    status = ps_info["overlap_status"]
    e_min  = ps_info["e_min"]
    e_max  = ps_info["e_max"]

    # choose outcome regressor
    outcome_model = str(outcome_model).lower()
    if outcome_model == "ridge":
        def make_reg():
            return Ridge(alpha=ridge_alpha)
    elif outcome_model in {"rf", "random_forest"}:
        def make_reg():
            return RandomForestRegressor(
                n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                max_depth=rf_max_depth,
                random_state=random_state
            )
    elif outcome_model in {"hgb", "hist_gb", "histgradientboosting"}:
        def make_reg():
            return HistGradientBoostingRegressor(
                max_depth=hgb_max_depth,
                learning_rate=hgb_learning_rate,
                max_iter=hgb_max_iter,
                random_state=random_state
            )
    else:
        def make_reg():
            return LinearRegression()

    # --- estimate mu0_hat, mu1_hat ---
    if len(x_cols) == 0:
        return {
            "estimator_name": "aipw_dr",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "n_used": int(n),
            "p_covariates": 0,
            "overlap_status": "violation",
            "e_min": float(e_min),
            "e_max": float(e_max),
            "notes": "AIPW/DR requires covariates X to fit mu1_hat(x), mu0_hat(x); no X-columns found."
        }
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
            "e_min": float(e_min),
            "e_max": float(e_max),
            "notes": "No treated or no control units after dropping missing rows."
        }

    # Fit separate outcome models
    m1 = make_reg()
    m0 = make_reg()
    m1.fit(X[idx1], Y[idx1])
    m0.fit(X[idx0], Y[idx0])

    mu1_hat = m1.predict(X).astype(float)
    mu0_hat = m0.predict(X).astype(float)

    # --- AIPW / DR score and estimate ---
    psi = (mu1_hat - mu0_hat) + (A * (Y - mu1_hat) / e_hat) - ((1 - A) * (Y - mu0_hat) / (1 - e_hat))

    ate_hat = float(np.mean(psi))
    se_hat = float(np.sqrt(np.var(psi, ddof=1) / n)) if n > 1 else None
    ci_95 = [ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat] if se_hat is not None else [None, None]

    notes = []
    notes.append(f"Outcome model: {outcome_model}")
    if outcome_model == "ridge":
        notes.append(f"ridge_alpha={ridge_alpha}")
    if outcome_model in {"rf", "random_forest"}:
        notes.append(f"rf_n_estimators={rf_n_estimators}, rf_min_samples_leaf={rf_min_samples_leaf}, rf_max_depth={rf_max_depth}")
    if outcome_model in {"hgb", "hist_gb", "histgradientboosting"}:
        notes.append(f"hgb_max_depth={hgb_max_depth}, hgb_learning_rate={hgb_learning_rate}, hgb_max_iter={hgb_max_iter}")
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
