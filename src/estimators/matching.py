"""Nearest-neighbor matching on the propensity score."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .propensity import fit_propensity_score
from .diagnostics import compute_ps_diagnostics
from .bootstrap import _bootstrap_estimates, _bootstrap_ci

def estimate_ps_matching(
    df: pd.DataFrame,
    ps_method: str = "logit",
    k_neighbors: int = 1,
    caliper: float = None,
    bootstrap_B: int = 0,
    seed: int = 0,
    ci_method: str = "percentile"
):
    fit = fit_propensity_score(df, method=ps_method)
    d, e_hat = fit["data"], fit["e_hat"]
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)

    diagnostics = compute_ps_diagnostics(d, e_hat, fit["raw_e_hat"])

    treated = np.where(A == 1)[0]
    control = np.where(A == 0)[0]
    if len(treated) == 0 or len(control) == 0:
        return {
            "estimator_name": f"ps_match_att_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "matching_diagnostics": None,
            "notes": "No treated or no control units."
        }

    k_eff = min(int(k_neighbors), len(control))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(e_hat[control].reshape(-1, 1))
    dist, idx = nn.kneighbors(e_hat[treated].reshape(-1, 1))

    keep = np.ones(len(treated), dtype=bool)
    if caliper is not None:
        keep = np.max(dist, axis=1) <= float(caliper)

    if np.sum(keep) == 0:
        return {
            "estimator_name": f"ps_match_att_{ps_method}",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "overlap_status": "violation",
            "overlap_warning": True,
            "diagnostics": diagnostics,
            "matching_diagnostics": None,
            "notes": "No matched treated units after caliper."
        }

    matched_controls = control[idx][keep]
    matched_y0 = np.mean(Y[matched_controls], axis=1)
    diffs = Y[treated[keep]] - matched_y0
    ate_hat = float(np.mean(diffs))

    match_diag = {
        "k_eff": int(k_eff),
        "match_rate": float(np.mean(keep)),
        "n_treated": int(len(treated)),
        "n_matched_treated": int(np.sum(keep)),
        "n_dropped_treated": int(np.sum(~keep))
    }

    se_hat, ci_95, note = None, [None, None], None
    if bootstrap_B > 0:
        def est_fn(d_b):
            fit_b = fit_propensity_score(d_b, method=ps_method)
            db, eb = fit_b["data"], fit_b["e_hat"]
            Ab = db["A"].to_numpy(dtype=int)
            Yb = db["Y"].to_numpy(dtype=float)

            treated_b = np.where(Ab == 1)[0]
            control_b = np.where(Ab == 0)[0]
            if len(treated_b) == 0 or len(control_b) == 0:
                return None

            k_b = min(int(k_neighbors), len(control_b))
            nn_b = NearestNeighbors(n_neighbors=k_b)
            nn_b.fit(eb[control_b].reshape(-1, 1))
            dist_b, idx_b = nn_b.kneighbors(eb[treated_b].reshape(-1, 1))

            keep_b = np.ones(len(treated_b), dtype=bool)
            if caliper is not None:
                keep_b = np.max(dist_b, axis=1) <= float(caliper)
            if np.sum(keep_b) == 0:
                return None

            matched_b = control_b[idx_b][keep_b]
            matched_y0_b = np.mean(Yb[matched_b], axis=1)
            return float(np.mean(Yb[treated_b[keep_b]] - matched_y0_b))

        boot = _bootstrap_estimates(d, est_fn, B=bootstrap_B, seed=seed)
        se_hat, ci_95, note = _bootstrap_ci(ate_hat, boot, ci_method=ci_method)

    return {
        "estimator_name": f"ps_match_att_{ps_method}",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": ci_95,
        "overlap_status": "warning" if diagnostics["overlap_warning"] else "ok",
        "overlap_warning": diagnostics["overlap_warning"],
        "diagnostics": diagnostics,
        "matching_diagnostics": match_diag,
        "notes": note
    }
