"""Propensity-score estimation and overlap diagnostics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def _prep_ps_rows(df: pd.DataFrame):
    if not {"Y", "A"}.issubset(df.columns):
        raise ValueError("df must contain columns 'Y' and 'A'.")

    x_cols = [c for c in df.columns if c.startswith("X")]
    use_cols = ["Y", "A"] + x_cols
    d = df[use_cols].copy()

    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d["A"] = pd.to_numeric(d["A"], errors="coerce").round()

    for c in x_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna()
    if d.shape[0] < 5:
        raise ValueError("Not enough complete rows after dropping missing values.")

    d["A"] = d["A"].astype(int).clip(0, 1)
    return d, x_cols

def estimate_propensity_logit(df: pd.DataFrame, clip: float = 1e-3):
    d, x_cols = _prep_ps_rows(df)
    n = d.shape[0]

    if len(x_cols) == 0:
        e_hat = np.full(n, float(d["A"].mean()))
    else:
        X = d[x_cols].to_numpy(dtype=float)
        A = d["A"].to_numpy(dtype=int)

        model = LogisticRegression(solver="liblinear", max_iter=2000)
        model.fit(X, A)
        e_hat = model.predict_proba(X)[:, 1]

    e_hat = np.clip(e_hat, clip, 1 - clip)
    return d, x_cols, e_hat

def overlap_status(e_hat: np.ndarray, warn=0.01, viol=0.001):
    e_min = float(np.min(e_hat)) if len(e_hat) else np.nan
    e_max = float(np.max(e_hat)) if len(e_hat) else np.nan
    if e_min <= viol or e_max >= 1 - viol:
        return "violation", e_min, e_max
    if e_min <= warn or e_max >= 1 - warn:
        return "warning", e_min, e_max
    return "ok", e_min, e_max
