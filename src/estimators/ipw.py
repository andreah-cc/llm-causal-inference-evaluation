"""IPW estimators: Horvitz-Thompson, Hájek/normalized, and stabilized IPW."""

import numpy as np
import pandas as pd

from .propensity import fit_propensity_score
from .diagnostics import compute_ps_diagnostics
from .bootstrap import _bootstrap_estimates, _bootstrap_ci

def estimate_ipw_ht(
    df: pd.DataFrame,
    ps_method: str = "logit",
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d, e_hat = fit["data"], fit["e_hat"]
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)

    diagnostics = compute_ps_diagnostics(d, e_hat, fit["raw_e_hat"])

    invalid_ps = np.any(~np.isfinite(e_hat)) or np.any(e_hat <= 0.0) or np.any(e_hat >= 1.0)
    if invalid_ps:
        return {
            "estimator_name": f"ipw_ht_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "notes": "HT-IPW is undefined because some estimated propensity scores are outside the open interval (0, 1)."
        }

    contrib = A * Y / e_hat - (1 - A) * Y / (1 - e_hat)
    ate_hat = float(np.mean(contrib))

    se_hat, ci_95, note = None, [None, None], None
    if bootstrap_B > 0:
        def est_fn(d_b):
            fit_b = fit_propensity_score(d_b, method=ps_method)
            db, eb = fit_b["data"], fit_b["e_hat"]
            Ab = db["A"].to_numpy(dtype=int)
            Yb = db["Y"].to_numpy(dtype=float)
            if np.any(~np.isfinite(eb)) or np.any(eb <= 0.0) or np.any(eb >= 1.0):
                return None
            return float(np.mean(Ab * Yb / eb - (1 - Ab) * Yb / (1 - eb)))

        boot = _bootstrap_estimates(d, est_fn, B=bootstrap_B, seed=seed)
        se_hat, ci_95, note = _bootstrap_ci(ate_hat, boot, ci_method=ci_method)

    return {
        "estimator_name": f"ipw_ht_{ps_method}",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": ci_95,
        "overlap_status": "warning" if diagnostics["overlap_warning"] else "ok",
        "overlap_warning": diagnostics["overlap_warning"],
        "diagnostics": diagnostics,
        "notes": note
    }

def estimate_ipw_hajek(
    df: pd.DataFrame,
    ps_method: str = "logit",
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d, e_hat = fit["data"], fit["e_hat"]
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)

    diagnostics = compute_ps_diagnostics(d, e_hat, fit["raw_e_hat"])

    invalid_ps = np.any(~np.isfinite(e_hat)) or np.any(e_hat <= 0.0) or np.any(e_hat >= 1.0)
    if invalid_ps:
        return {
            "estimator_name": f"ipw_hajek_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "notes": "Hájek IPW is undefined because some estimated propensity scores are outside the open interval (0, 1)."
        }

    num1 = np.sum(A * Y / e_hat)
    den1 = np.sum(A / e_hat)
    num0 = np.sum((1 - A) * Y / (1 - e_hat))
    den0 = np.sum((1 - A) / (1 - e_hat))
    ate_hat = float(num1 / den1 - num0 / den0)

    se_hat, ci_95, note = None, [None, None], None
    if bootstrap_B > 0:
        def est_fn(d_b):
            fit_b = fit_propensity_score(d_b, method=ps_method)
            db, eb = fit_b["data"], fit_b["e_hat"]
            Ab = db["A"].to_numpy(dtype=int)
            Yb = db["Y"].to_numpy(dtype=float)
            if np.any(~np.isfinite(eb)) or np.any(eb <= 0.0) or np.any(eb >= 1.0):
                return None
            num1b = np.sum(Ab * Yb / eb)
            den1b = np.sum(Ab / eb)
            num0b = np.sum((1 - Ab) * Yb / (1 - eb))
            den0b = np.sum((1 - Ab) / (1 - eb))
            return float(num1b / den1b - num0b / den0b)

        boot = _bootstrap_estimates(d, est_fn, B=bootstrap_B, seed=seed)
        se_hat, ci_95, note = _bootstrap_ci(ate_hat, boot, ci_method=ci_method)

    return {
        "estimator_name": f"ipw_hajek_{ps_method}",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": ci_95,
        "overlap_status": "warning" if diagnostics["overlap_warning"] else "ok",
        "overlap_warning": diagnostics["overlap_warning"],
        "diagnostics": diagnostics,
        "notes": note
    }

def estimate_ipw_stabilized(
    df: pd.DataFrame,
    ps_method: str = "logit",
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d, e_hat = fit["data"], fit["e_hat"]
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)

    diagnostics = compute_ps_diagnostics(d, e_hat, fit["raw_e_hat"])

    invalid_ps = np.any(~np.isfinite(e_hat)) or np.any(e_hat <= 0.0) or np.any(e_hat >= 1.0)
    if invalid_ps:
        return {
            "estimator_name": f"ipw_stabilized_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "notes": "Stabilized IPW is undefined because some estimated propensity scores are outside the open interval (0, 1)."
        }

    p_hat = float(np.mean(A))
    w1 = p_hat * A / e_hat
    w0 = (1 - p_hat) * (1 - A) / (1 - e_hat)
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    ate_hat = float(mu1 - mu0)

    se_hat, ci_95, note = None, [None, None], None
    if bootstrap_B > 0:
        def est_fn(d_b):
            fit_b = fit_propensity_score(d_b, method=ps_method)
            db, eb = fit_b["data"], fit_b["e_hat"]
            Ab = db["A"].to_numpy(dtype=int)
            Yb = db["Y"].to_numpy(dtype=float)
            if np.any(~np.isfinite(eb)) or np.any(eb <= 0.0) or np.any(eb >= 1.0):
                return None
            pb = float(np.mean(Ab))
            w1b = pb * Ab / eb
            w0b = (1 - pb) * (1 - Ab) / (1 - eb)
            mu1b = np.sum(w1b * Yb) / np.sum(w1b)
            mu0b = np.sum(w0b * Yb) / np.sum(w0b)
            return float(mu1b - mu0b)

        boot = _bootstrap_estimates(d, est_fn, B=bootstrap_B, seed=seed)
        se_hat, ci_95, note = _bootstrap_ci(ate_hat, boot, ci_method=ci_method)

    return {
        "estimator_name": f"ipw_stabilized_{ps_method}",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": ci_95,
        "overlap_status": "warning" if diagnostics["overlap_warning"] else "ok",
        "overlap_warning": diagnostics["overlap_warning"],
        "diagnostics": diagnostics,
        "notes": note
    }
