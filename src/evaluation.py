"""Evaluation metrics for LLM ATE estimates."""

import numpy as np


def point_estimation(results):
    tau_true = np.asarray(results["tau_true"], dtype=float)
    tau_hat = np.asarray(results["ate_hat"], dtype=float)
    bias = np.mean(tau_hat - tau_true)
    rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    return bias, rmse


def compute_uncertainty(results):
    tau_true = np.asarray(results["tau_true"], dtype=float)
    lower = np.asarray(results["ci95_lower"], dtype=float)
    upper = np.asarray(results["ci95_upper"], dtype=float)
    coverage = np.mean((tau_true >= lower) & (tau_true <= upper))
    avg_width = np.mean(upper - lower)
    return coverage, avg_width
