"""Regression-adjustment plug-in baseline."""

import numpy as np
import statsmodels.api as sm

from .propensity import estimate_propensity_logit, overlap_status

def baseline_ra_plugin(
    df,
    robust=True,
    ci_level=0.95,
    clip=1e-3
):

    # propensity score and overlap check
    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)

    # extract treatment and outcome
    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]

    status, e_min, e_max = overlap_status(e_hat)

    # when sample size is samll, return empty
    if n < 5:
        return {
            "estimator_name": "ra_plugin",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "n_used": int(n),
            "p_covariates": int(len(x_cols)),
            "overlap_status": "violation",
            "e_min": e_min,
            "e_max": e_max,
            "notes": "Too few observations after preprocessing."
        }

    # no covariates
    if len(x_cols) == 0:
        if not np.any(A == 1) or not np.any(A == 0):
            return {
                "estimator_name": "ra_plugin",
                "ate_hat": None,
                "se_hat": None,
                "ci_95": [None, None],
                "n_used": int(n),
                "p_covariates": 0,
                "overlap_status": "violation",
                "e_min": e_min,
                "e_max": e_max,
                "notes": "No treated or no control units."
            }

        mu1 = float(np.mean(Y[A == 1]))
        mu0 = float(np.mean(Y[A == 0]))

        mu1_hat = np.full(n, mu1)
        mu0_hat = np.full(n, mu0)

        d_hat = mu1_hat - mu0_hat
        ate_hat = float(np.mean(d_hat))
        se_hat = float(np.sqrt(np.var(d_hat, ddof=1) / n)) if n > 1 else None

    # with covariates
    else:
        X = d[x_cols].to_numpy(dtype=float)

        # construct regression design matrix: [1, A, X]
        X_design = sm.add_constant(
            np.column_stack([A, X]),
            has_constant="add"
        )

        fit = sm.OLS(Y, X_design).fit()
        if robust:
            fit = fit.get_robustcov_results(cov_type="HC1")

        X1 = X_design.copy()
        X1[:, 1] = 1

        X0 = X_design.copy()
        X0[:, 1] = 0

        mu1_hat = fit.predict(X1)
        mu0_hat = fit.predict(X0)

        d_hat = mu1_hat - mu0_hat
        ate_hat = float(np.mean(d_hat))

        # use delta method for SE
        covb = np.asarray(fit.cov_params())
        diff_design = (X1 - X0)
        grad = diff_design.mean(axis=0).reshape(-1, 1)

        var_ate = float(grad.T @ covb @ grad)
        se_hat = np.sqrt(var_ate) if np.isfinite(var_ate) and var_ate >= 0 else None

    # CI
    alpha = 1 - ci_level
    z = 1.96
    if abs(ci_level - 0.95) > 1e-12:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))

    ci_95 = [ate_hat - z * se_hat, ate_hat + z * se_hat] if se_hat is not None else [None, None]

    # diagnostics
    notes = []
    if robust:
        notes.append("HC1 robust SE.")
    if status != "ok":
        notes.append(f"Overlap {status}: e_hat in [{e_min:.4g}, {e_max:.4g}].")

    return {
        "estimator_name": "ra_plugin",
        "ate_hat": ate_hat,
        "se_hat": se_hat,
        "ci_95": [float(ci_95[0]), float(ci_95[1])] if se_hat is not None else ci_95,
        "n_used": int(n),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": " ".join(notes) if notes else None
    }
