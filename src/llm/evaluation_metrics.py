"""Module 4 evaluation metrics for point estimates and intervals."""

import numpy as np


def get_tau_from_custom_id(custom_id):
    try:
        return float(custom_id.split("_")[1])
    except (IndexError, ValueError, AttributeError):
        return None


def _extract_ci(record):
    """Read CI from either ci95_lower/ci95_upper or the structured ci_95 field."""
    lower = record.get("ci95_lower")
    upper = record.get("ci95_upper")

    if lower is None or upper is None:
        ci = record.get("ci_95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lower, upper = ci

    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError):
        return None

    if not (np.isfinite(lower) and np.isfinite(upper)):
        return None

    return lower, upper


def point_estimation(result):
    errors = []

    for record in result:
        tau_true = get_tau_from_custom_id(record.get("custom_id"))
        ate_hat = record.get("ate_hat")

        if ate_hat is None or tau_true is None:
            continue

        try:
            errors.append(float(ate_hat) - tau_true)
        except (TypeError, ValueError):
            continue

    if not errors:
        return {
            "n_valid": 0,
            "bias": np.nan,
            "RMSE": np.nan,
            "median absolute error": np.nan,
        }

    errors = np.asarray(errors, dtype=float)

    return {
        "n_valid": int(errors.size),
        "bias": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "median absolute error": float(np.median(np.abs(errors))),
    }


def coverage(result):
    covered = []
    widths = []

    for record in result:
        if record.get("format_failure") is True:
            continue

        tau_true = get_tau_from_custom_id(record.get("custom_id"))
        ci = _extract_ci(record)

        if tau_true is None or ci is None:
            continue

        lower, upper = ci
        covered.append(lower <= tau_true <= upper)
        widths.append(upper - lower)

    if not widths:
        return {
            "n_ci": 0,
            "nominal level": 0.95,
            "coverage": np.nan,
            "avg_width": np.nan,
        }

    return {
        "n_ci": int(len(widths)),
        "nominal level": 0.95,
        "coverage": float(np.mean(covered)),
        "avg_width": float(np.mean(widths)),
    }


def width_vs_coverage_tradeoff(result, n_bins=5):
    """Bin intervals by width and report empirical coverage within each bin.

    This fixes a bug in the uploaded notebook where `covered` was stored as a
    single count and then indexed as if it were an array.
    """
    widths = []
    covered = []

    for record in result:
        if record.get("format_failure") is True:
            continue

        tau_true = get_tau_from_custom_id(record.get("custom_id"))
        ci = _extract_ci(record)

        if tau_true is None or ci is None:
            continue

        lower, upper = ci
        widths.append(upper - lower)
        covered.append(float(lower <= tau_true <= upper))

    widths = np.asarray(widths, dtype=float)
    covered = np.asarray(covered, dtype=float)

    if widths.size == 0:
        return {"n_ci": 0, "bins": []}

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(widths, quantiles))

    bins = []
    for i in range(len(edges) - 1):
        lower_edge = edges[i]
        upper_edge = edges[i + 1]

        if i == len(edges) - 2:
            mask = (widths >= lower_edge) & (widths <= upper_edge)
        else:
            mask = (widths >= lower_edge) & (widths < upper_edge)

        if not np.any(mask):
            continue

        bins.append(
            {
                "bin": f"[{lower_edge:.3g}, {upper_edge:.3g}]",
                "n": int(mask.sum()),
                "avg_width": float(np.mean(widths[mask])),
                "coverage": float(np.mean(covered[mask])),
            }
        )

    return {
        "n_ci": int(widths.size),
        "bins": bins,
    }
