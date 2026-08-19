"""Module 3 ICL input builders for Stage 3, Stage 4, and Stage 5."""

import numpy as np
import pandas as pd

from src.estimators.diagnostics import build_ps_bin_summary, compute_ps_diagnostics

def build_protocol_c_input(d: pd.DataFrame, e_hat: np.ndarray, K: int = 5):
    """
    Stage 3 input:
    Build PS-bin summaries for stratification prompt.
    Here d should be the cleaned data returned by fit_propensity_score(... )["data"].
    """
    bin_info = build_ps_bin_summary(d, e_hat, K=K)

    return {
        "K": int(bin_info["K_eff"]),
        "bin_summary": bin_info["bin_summary"],
        "empty_bin_flag": bool(bin_info["empty_bin_flag"]),
        "assumptions": ["unconfoundedness_given_X", "overlap"]
    }

def build_protocol_ipw_input(d: pd.DataFrame, e_hat: np.ndarray):
    """
    Stage 4 input:
    Build aggregated sufficient statistics for Hájek IPW prompt.
    """
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    e_hat = np.asarray(e_hat, dtype=float)

    # safety clip
    e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

    S1Y = float(np.sum(A * Y / e_hat))
    S1  = float(np.sum(A / e_hat))
    S0Y = float(np.sum((1 - A) * Y / (1 - e_hat)))
    S0  = float(np.sum((1 - A) / (1 - e_hat)))

    diagnostics = compute_ps_diagnostics(d, e_hat, raw_e_hat=e_hat)

    wd = diagnostics["weight_diagnostics"]

    return {
        "S1Y": S1Y,
        "S1": S1,
        "S0Y": S0Y,
        "S0": S0,
        "max_weight": wd["max_weight"],
        "ESS_treated": wd["ESS_treated"],
        "ESS_control": wd["ESS_control"],
        "assumptions": ["unconfoundedness_given_X", "overlap"]
    }

def build_stage5_input(
    d: pd.DataFrame,
    e_hat: np.ndarray,
    candidate_estimates: dict = None,
    K: int = 5
):
    """
    Stage 5 input:
    Build diagnostics + optional candidate estimates for estimator-choice prompt.
    """
    A = d["A"].to_numpy(dtype=int)
    e_hat = np.asarray(e_hat, dtype=float)
    e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

    diagnostics = compute_ps_diagnostics(d, e_hat, raw_e_hat=e_hat)
    c_input = build_protocol_c_input(d, e_hat, K=K)

    pq = diagnostics["propensity_quantiles"]
    wd = diagnostics["weight_diagnostics"]

    return {
        "propensity_treated_quantiles": pq["treated"],
        "propensity_control_quantiles": pq["control"],
        "max_weight": wd["max_weight"],
        "ESS_treated": wd["ESS_treated"],
        "ESS_control": wd["ESS_control"],
        "empty_bins": c_input["empty_bin_flag"],
        "candidate_estimates": candidate_estimates or {}
    }
