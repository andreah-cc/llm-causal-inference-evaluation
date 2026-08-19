"""Point-estimation and uncertainty-calibration metrics."""

import numpy as np

# point estimation metrics
def point_estimation(results):
    tau_true = np.array(results['tau_true'], dtype=float)
    tau_hat = np.array(results['ate_hat'], dtype=float)

    valid = np.isfinite(tau_hat)
    if valid.any():
        bias = np.mean(tau_hat[valid] - tau_true[valid])
        rmse = np.sqrt(np.mean((tau_hat[valid] - tau_true[valid]) ** 2))
    else:
        bias = np.nan
        rmse = np.nan
    return bias, rmse

# uncertainty calibration
def compute_uncertainty(results):
    tau_true = np.array(results['tau_true'], dtype=float)
    L = np.array(results['ci95_lower'], dtype=float)
    U = np.array(results['ci95_upper'], dtype=float)

    valid = np.isfinite(L) & np.isfinite(U)
    if valid.any():
        coverage = np.mean((tau_true[valid] >= L[valid]) & (tau_true[valid] <= U[valid]))
        avg_width = np.mean(U[valid] - L[valid])
    else:
        coverage = np.nan
        avg_width = np.nan

    missing_ci_rate = 1.0 - np.mean(valid)
    return coverage, avg_width, missing_ci_rate
