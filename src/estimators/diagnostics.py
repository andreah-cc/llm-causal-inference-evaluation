"""Overlap, weight-stability, and propensity-bin diagnostics."""

import numpy as np
import pandas as pd

def _q(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"min": None, "q05": None, "q50": None, "q95": None, "max": None}
    return {
        "min": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x))
    }

def _ess(w):
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return None
    denom = np.sum(w ** 2)
    if denom <= 0:
        return None
    return float((np.sum(w) ** 2) / denom)

def _safe_group_weights(A: np.ndarray, e_hat: np.ndarray):
    A = np.asarray(A, dtype=int)
    e_hat = np.asarray(e_hat, dtype=float)

    treated_mask = (A == 1)
    control_mask = (A == 0)

    e_treated = e_hat[treated_mask]
    e_control = e_hat[control_mask]

    w_treated = np.full(len(e_treated), np.inf, dtype=float)
    ok_treated = np.isfinite(e_treated) & (e_treated > 0)
    w_treated[ok_treated] = 1.0 / e_treated[ok_treated]

    w_control = np.full(len(e_control), np.inf, dtype=float)
    ok_control = np.isfinite(e_control) & (e_control < 1)
    w_control[ok_control] = 1.0 / (1.0 - e_control[ok_control])

    return w_treated, w_control

def compute_ps_diagnostics(
    d: pd.DataFrame,
    e_hat: np.ndarray,
    raw_e_hat: np.ndarray = None,
    bins: int = 10,
    max_weight_warn: float = 20.0,
    ess_warn: float = 50.0
):
    A = d["A"].to_numpy(dtype=int)
    e_hat = np.asarray(e_hat, dtype=float)
    raw_e_hat = e_hat if raw_e_hat is None else np.asarray(raw_e_hat, dtype=float)

    w_treated, w_control = _safe_group_weights(A, e_hat)
    all_w = np.concatenate([w_treated, w_control]) if len(w_treated) + len(w_control) > 0 else np.array([])

    max_weight = float(np.max(all_w)) if len(all_w) else None
    p99_weight = float(np.quantile(all_w, 0.99)) if len(all_w) else None
    ess_treated = _ess(w_treated)
    ess_control = _ess(w_control)

    invalid_ps = bool(
        np.any(~np.isfinite(e_hat)) or np.any(e_hat <= 0.0) or np.any(e_hat >= 1.0)
    )

    flags = []
    if invalid_ps:
        flags.append("propensity_outside_(0,1)")
    if max_weight is not None and max_weight > max_weight_warn:
        flags.append(f"max_weight>{max_weight_warn}")
    if ess_treated is not None and ess_treated < ess_warn:
        flags.append(f"ESS_treated<{ess_warn}")
    if ess_control is not None and ess_control < ess_warn:
        flags.append(f"ESS_control<{ess_warn}")

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    hist_all, _ = np.histogram(e_hat[np.isfinite(e_hat)], bins=bin_edges)
    hist_treated, _ = np.histogram(e_hat[(A == 1) & np.isfinite(e_hat)], bins=bin_edges)
    hist_control, _ = np.histogram(e_hat[(A == 0) & np.isfinite(e_hat)], bins=bin_edges)

    return {
        "overlap_status": "warning" if flags else "ok",
        "overlap_warning": bool(flags),
        "warning_flags": flags,
        "propensity_quantiles": {
            "all": _q(e_hat),
            "treated": _q(e_hat[A == 1]),
            "control": _q(e_hat[A == 0]),
            "raw_all": _q(raw_e_hat)
        },
        "e_min": float(np.min(e_hat[np.isfinite(e_hat)])) if np.any(np.isfinite(e_hat)) else None,
        "e_max": float(np.max(e_hat[np.isfinite(e_hat)])) if np.any(np.isfinite(e_hat)) else None,
        "weight_diagnostics": {
            "max_weight": max_weight,
            "p99_weight": p99_weight,
            "ESS_treated": ess_treated,
            "ESS_control": ess_control
        },
        "histogram": {
            "bin_edges": bin_edges.tolist(),
            "all": hist_all.tolist(),
            "treated": hist_treated.tolist(),
            "control": hist_control.tolist()
        }
    }

def build_ps_bin_summary(d: pd.DataFrame, e_hat: np.ndarray, K: int = 5):
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = len(d)

    bins = pd.qcut(pd.Series(e_hat), q=K, labels=False, duplicates="drop")
    if bins.isna().all():
        bins = pd.Series(np.zeros(n, dtype=int))
        K_eff = 1
    else:
        K_eff = int(bins.max()) + 1

    out = []
    empty_bin_flag = False

    for b in range(K_eff):
        idx = (bins.to_numpy() == b)
        y1 = Y[idx & (A == 1)]
        y0 = Y[idx & (A == 0)]
        n1, n0 = len(y1), len(y0)
        nb = int(np.sum(idx))

        empty = (n1 == 0 or n0 == 0)
        if empty:
            empty_bin_flag = True

        out.append({
            "bin": int(b),
            "n": nb,
            "n1": int(n1),
            "n0": int(n0),
            "weight": float(nb / n),
            "meanY1": float(np.mean(y1)) if n1 > 0 else None,
            "sdY1": float(np.std(y1, ddof=1)) if n1 > 1 else None,
            "meanY0": float(np.mean(y0)) if n0 > 0 else None,
            "sdY0": float(np.std(y0, ddof=1)) if n0 > 1 else None,
            "empty_support": bool(empty)
        })

    return {
        "K_requested": int(K),
        "K_eff": int(K_eff),
        "empty_bin_flag": bool(empty_bin_flag),
        "bin_summary": out
    }
