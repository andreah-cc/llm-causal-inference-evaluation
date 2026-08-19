"""Compatibility helpers for two names referenced but not defined in the uploaded Module 3 notebook.

These wrappers do not introduce a new estimator. They adapt `fit_propensity_score`
and the notebook's own overlap diagnostics to the tuple/dictionary interfaces
expected by `ra_plugin` and `baseline_aipw_dr`.
"""

import numpy as np

from .propensity import fit_propensity_score
from .diagnostics import compute_ps_diagnostics


def estimate_propensity_logit(df, clip=1e-3):
    fit = fit_propensity_score(df, method="logit")
    d = fit["data"]
    x_cols = fit["x_cols"]
    raw = np.asarray(fit["e_hat"], dtype=float)
    e_hat = np.clip(raw, clip, 1.0 - clip)

    diagnostics = compute_ps_diagnostics(d, e_hat, raw_e_hat=raw)
    status = "warning" if diagnostics["overlap_warning"] else "ok"

    ps_info = {
        "overlap_status": status,
        "e_min": float(np.min(e_hat)),
        "e_max": float(np.max(e_hat)),
        "diagnostics": diagnostics,
    }
    return d, x_cols, e_hat, ps_info


def overlap_status(e_hat):
    e_hat = np.asarray(e_hat, dtype=float)
    if len(e_hat) == 0 or np.any(~np.isfinite(e_hat)):
        return "violation", None, None

    e_min = float(np.min(e_hat))
    e_max = float(np.max(e_hat))

    # Use the Module 3 diagnostics to determine whether weight/overlap behavior
    # is concerning. Since this helper receives only e_hat, boundary violations
    # can be detected directly; non-boundary warning logic is handled upstream.
    if e_min <= 0.0 or e_max >= 1.0:
        return "violation", e_min, e_max
    return "ok", e_min, e_max
