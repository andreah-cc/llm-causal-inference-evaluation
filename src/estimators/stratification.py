"""Propensity-score stratification estimator."""

import numpy as np
import pandas as pd

from .propensity import fit_propensity_score
from .diagnostics import compute_ps_diagnostics, build_ps_bin_summary
from .bootstrap import _bootstrap_estimates, _bootstrap_ci

def estimate_ps_stratification(
    df: pd.DataFrame,
    ps_method: str = "logit",
    K: int = 5,
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d, e_hat = fit["data"], fit["e_hat"]
    diagnostics = compute_ps_diagnostics(d, e_hat, fit["raw_e_hat"])
    bin_info = build_ps_bin_summary(d, e_hat, K=K)

    if bin_info["empty_bin_flag"]:
        return {
            "estimator_name": f"ps_strat_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "bin_info": bin_info,
            "notes": "At least one bin has n1=0 or n0=0."
        }

    ate_hat = float(sum(
        b["weight"] * (b["meanY1"] - b["meanY0"])
        for b in bin_info["bin_summary"]
    ))

    se_hat, ci_95, note = None, [None, None], None
    if bootstrap_B > 0:
        def est_fn(d_b):
            fit_b = fit_propensity_score(d_b, method=ps_method)
            bin_b = build_ps_bin_summary(fit_b["data"], fit_b["e_hat"], K=K)
            if bin_b["empty_bin_flag"]:
                return None
            return float(sum(
                b["weight"] * (b["meanY1"] - b["meanY0"])
                for b in bin_b["bin_summary"]
            ))

        boot = _bootstrap_estimates(d, est_fn, B=bootstrap_B, seed=seed)
        se_hat, ci_95, note = _bootstrap_ci(ate_hat, boot, ci_method=ci_method)

    return {
        "estimator_name": f"ps_strat_{ps_method}",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": ci_95,
        "overlap_status": "warning" if diagnostics["overlap_warning"] else "ok",
        "overlap_warning": diagnostics["overlap_warning"],
        "diagnostics": diagnostics,
        "bin_info": bin_info,
        "notes": note
    }
