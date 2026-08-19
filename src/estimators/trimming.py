"""Common-support restriction / trimming plus Hájek IPW."""

import numpy as np
import pandas as pd

from .propensity import fit_propensity_score
from .diagnostics import compute_ps_diagnostics
from .bootstrap import _bootstrap_estimates, _bootstrap_ci

def apply_common_support_restriction(
    d: pd.DataFrame,
    e_hat: np.ndarray,
    mode: str = "common_support",
    lower: float = None,
    upper: float = None
):
    A = d["A"].to_numpy(dtype=int)
    e_hat = np.asarray(e_hat, dtype=float)

    if mode == "common_support":
        lo = max(float(np.min(e_hat[A == 1])), float(np.min(e_hat[A == 0])))
        hi = min(float(np.max(e_hat[A == 1])), float(np.max(e_hat[A == 0])))
    elif mode == "trim":
        if lower is None or upper is None:
            raise ValueError("For mode='trim', lower and upper must be provided.")
        lo, hi = float(lower), float(upper)
    else:
        raise ValueError("mode must be 'common_support' or 'trim'.")

    keep = (e_hat >= lo) & (e_hat <= hi)

    return {
        "trim_mode": mode,
        "lower": lo,
        "upper": hi,
        "keep_mask": keep,
        "n_before": int(len(d)),
        "n_after": int(np.sum(keep)),
        "n_dropped": int(np.sum(~keep))
    }

def estimate_trimmed_ipw_hajek(
    df: pd.DataFrame,
    ps_method: str = "logit",
    trim_mode: str = "common_support",
    trim_lower: float = None,
    trim_upper: float = None,
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d_full, e_hat_full = fit["data"], fit["e_hat"]

    pre_trim_diagnostics = compute_ps_diagnostics(d_full, e_hat_full, fit["raw_e_hat"])
    trim_info = apply_common_support_restriction(
        d_full,
        e_hat_full,
        mode=trim_mode,
        lower=trim_lower,
        upper=trim_upper
    )

    keep = trim_info["keep_mask"]
    d_trim = d_full.loc[keep].reset_index(drop=True)

    if len(d_trim) < 5 or d_trim["A"].nunique() < 2:
        return {
            "estimator_name": f"trimmed_ipw_hajek_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "trim_info": {k: v for k, v in trim_info.items() if k != "keep_mask"},
            "pre_trim_diagnostics": pre_trim_diagnostics,
            "post_trim_diagnostics": None,
            "notes": "After trimming/common-support restriction, too few observations or only one treatment arm remained."
        }

    trimmed_res = estimate_ipw_hajek(
        d_trim,
        ps_method=ps_method,
        bootstrap_B=bootstrap_B,
        seed=seed,
        ci_method=ci_method
    )

    notes = []
    if trimmed_res.get("notes"):
        notes.append(trimmed_res["notes"])
    notes.append(
        "Estimand after trimming pertains to the retained overlap region rather than the full original sample."
    )

    return {
        "estimator_name": f"trimmed_ipw_hajek_{ps_method}",
        "ate_hat": trimmed_res["ate_hat"],
        "se_hat": trimmed_res["se_hat"],
        "ci_95": trimmed_res["ci_95"],
        "overlap_status": trimmed_res["overlap_status"],
        "overlap_warning": trimmed_res["overlap_warning"],
        "trim_info": {k: v for k, v in trim_info.items() if k != "keep_mask"},
        "pre_trim_diagnostics": pre_trim_diagnostics,
        "post_trim_diagnostics": trimmed_res["diagnostics"],
        "notes": " ".join(notes)
    }
