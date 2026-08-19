"""ICL stress tests / failure-mode checks.

These functions are based on the Module 2 notebook, with notebook-global
dependencies replaced by explicit function arguments so they can be reused
from a package.
"""

import numpy as np
import pandas as pd

from src.simulation.common import sigmoid
from src.simulation.engine5 import main_5


def trap_1(csv_path, rng=None):
    """Add post-treatment Z to test whether the LLM is tempted to adjust for it."""
    if rng is None:
        rng = np.random.default_rng()

    df = pd.read_csv(csv_path)
    x_cols = [c for c in df.columns if c.startswith("X")]
    p = len(x_cols)
    n_rows = len(df)

    lambd = 1.0
    coef = rng.normal(loc=0.0, scale=0.2, size=p)
    noise = rng.normal(loc=0.0, scale=0.2, size=n_rows)

    df["Z"] = (
        lambd * df["A"].values
        + (coef.T @ df[x_cols].values.T)
        + noise
    )
    return df


def trap_2(csv_path, rng=None):
    """Add V that strongly predicts treatment but does not directly affect Y."""
    if rng is None:
        rng = np.random.default_rng()

    df = pd.read_csv(csv_path)
    rho = 2.0

    x_cols = [c for c in df.columns if c.startswith("X")]
    n_rows = df.shape[0]

    V = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    df["V"] = V

    beta = np.ones(len(x_cols)) / np.sqrt(len(x_cols))
    X_matrix = df[x_cols].values
    linear_score = X_matrix @ beta + rho * V

    df["A"] = rng.binomial(1, sigmoid(linear_score))
    return df


def trap_3(X, w, a, k, t, engine, *, eps, b=None, SY=None, j_star=None, ep=0.01, rng=None):
    """Create a weak-overlap case using Engine 5."""
    return main_5(
        X, w, a, k, ep, t, engine,
        eps=eps, b=b, SY=SY, j_star=j_star, rng=rng
    )


def check_warning(llm_output, weak_overlap: bool):
    status = llm_output.get("overlap_assessment", {}).get("status", "unknown")
    predicted_weak = status in ["warning", "violation"]
    return predicted_weak == weak_overlap


def check_stabilization_suggestion(llm_output):
    texts = []
    roc = llm_output.get("refusal_or_caution")
    if isinstance(roc, str):
        texts.append(roc.lower())

    oa = llm_output.get("overlap_assessment", {})
    if isinstance(oa, dict) and isinstance(oa.get("notes"), str):
        texts.append(oa["notes"].lower())

    text = " ".join(texts)

    keywords = [
        "trim", "trimming", "truncate", "truncation",
        "stabilized", "stabilization",
        "weight", "overlap", "positivity"
    ]
    return any(k in text for k in keywords)


def interval_width(llm_output):
    ci = llm_output.get("ci_95")
    if not (isinstance(ci, list) and len(ci) == 2):
        return np.nan
    try:
        L = float(ci[0]) if ci[0] is not None else np.nan
        U = float(ci[1]) if ci[1] is not None else np.nan
    except Exception:
        return np.nan
    if np.isfinite(L) and np.isfinite(U):
        return U - L
    return np.nan
