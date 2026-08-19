"""Regression-adjustment plug-in estimator (OLS or neural-network outcome model)."""

import numpy as np
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .compat import estimate_propensity_logit, overlap_status
from .bootstrap import _bootstrap_ate

def ra_plugin(
    df,
    robust=True,
    ci_level=0.95,
    clip=1e-3,
    use_nn=False,
    hidden_layer_sizes=(64, 64),
    alpha=1e-4,
    max_iter=2000,
    early_stopping=True,
    B=300,
    seed=123
):

    d, x_cols, e_hat = estimate_propensity_logit(df, clip=clip)

    A = d["A"].to_numpy(dtype=int)
    Y = d["Y"].to_numpy(dtype=float)
    n = d.shape[0]

    status, e_min, e_max = overlap_status(e_hat)

    if n < 5:
        return {
            "estimator_name": "ra_plugin_nn" if use_nn else "ra_plugin_ols",
            "ate_hat": None,
            "se_hat": None,
            "ci_95": [None, None],
            "n_used": int(n),
            "p_covariates": int(len(x_cols)),
            "overlap_status": "violation",
            "e_min": e_min,
            "e_max": e_max,
            "notes": "Too few observations."
        }

    # CASE 1: No covariates
    if len(x_cols) == 0:
        if not np.any(A == 1) or not np.any(A == 0):
            return {
                "estimator_name": "diff_in_means",
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
        ate_hat = mu1 - mu0

        # bootstrap SE / CI
        def ate_fn_no_x(d_boot):
            Ab = d_boot["A"].to_numpy(dtype=int)
            Yb = d_boot["Y"].to_numpy(dtype=float)

            if not np.any(Ab == 1) or not np.any(Ab == 0):
                return np.nan

            mu1b = np.mean(Yb[Ab == 1])
            mu0b = np.mean(Yb[Ab == 0])
            return float(mu1b - mu0b)

        se_hat, ci_95 = _bootstrap_ate(d, ate_fn_no_x, B=B, seed=seed, ci_level=ci_level)

        notes = [
            "Difference-in-means estimator (no covariates).",
            f"Bootstrap SE/CI with B={B}."
        ]
        if status != "ok":
            notes.append(f"Overlap {status}: e_hat in [{e_min:.4g}, {e_max:.4g}].")

        return {
            "estimator_name": "diff_in_means",
            "ate_hat": float(ate_hat),
            "se_hat": float(se_hat),
            "ci_95": [float(ci_95[0]), float(ci_95[1])],
            "n_used": int(n),
            "p_covariates": 0,
            "overlap_status": status,
            "e_min": e_min,
            "e_max": e_max,
            "notes": " ".join(notes)
        }

    # CASE 2: Have covariates
    X = d[x_cols].to_numpy(dtype=float)

    # METHOD A: OLS RA baseline
    if not use_nn:
        # Fit on original sample
        Z = sm.add_constant(np.column_stack([A, X]), has_constant="add")
        fit = sm.OLS(Y, Z).fit()

        Z1 = Z.copy()
        Z1[:, 1] = 1
        Z0 = Z.copy()
        Z0[:, 1] = 0

        mu1_hat = fit.predict(Z1)
        mu0_hat = fit.predict(Z0)

        d_hat = mu1_hat - mu0_hat
        ate_hat = float(np.mean(d_hat))

        # bootstrap SE / CI for OLS plug-in ATE
        def ate_fn_ols(d_boot):
            Ab = d_boot["A"].to_numpy(dtype=int)
            Yb = d_boot["Y"].to_numpy(dtype=float)
            Xb = d_boot[x_cols].to_numpy(dtype=float)

            Zb = sm.add_constant(np.column_stack([Ab, Xb]), has_constant="add")
            fit_b = sm.OLS(Yb, Zb).fit()

            Z1b = Zb.copy()
            Z1b[:, 1] = 1
            Z0b = Zb.copy()
            Z0b[:, 1] = 0

            mu1b = fit_b.predict(Z1b)
            mu0b = fit_b.predict(Z0b)

            return float(np.mean(mu1b - mu0b))

        se_hat, ci_95 = _bootstrap_ate(d, ate_fn_ols, B=B, seed=seed, ci_level=ci_level)

        notes = [
            "Outcome regression plug-in estimator (OLS).",
            "ATE computed as mean(m_hat(1,X) - m_hat(0,X)).",
            f"Bootstrap SE/CI with B={B}.",
            "OLS outcome model: Y ~ A + X."
        ]
        if robust:
            notes.append("Robust option not used for SE/CI because inference is bootstrap-based.")
        if status != "ok":
            notes.append(f"Overlap {status}: e_hat in [{e_min:.4g}, {e_max:.4g}].")

        return {
            "estimator_name": "ra_plugin_ols",
            "ate_hat": float(ate_hat),
            "se_hat": float(se_hat),
            "ci_95": [float(ci_95[0]), float(ci_95[1])],
            "n_used": int(n),
            "p_covariates": int(len(x_cols)),
            "overlap_status": status,
            "e_min": e_min,
            "e_max": e_max,
            "notes": " ".join(notes)
        }

    # METHOD B: NN baseline
    W = np.column_stack([A, X])

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=seed
        ))
    ])

    model.fit(W, Y)

    W1 = W.copy()
    W1[:, 0] = 1
    W0 = W.copy()
    W0[:, 0] = 0

    mu1_hat = model.predict(W1)
    mu0_hat = model.predict(W0)

    d_hat = mu1_hat - mu0_hat
    ate_hat = float(np.mean(d_hat))

    def ate_fn(d_boot):
        Ab = d_boot["A"].to_numpy(dtype=int)
        Yb = d_boot["Y"].to_numpy(dtype=float)
        Xb = d_boot[x_cols].to_numpy(dtype=float)

        Wb = np.column_stack([Ab, Xb])

        mb = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                early_stopping=early_stopping,
                random_state=seed
            ))
        ])
        mb.fit(Wb, Yb)

        W1b = Wb.copy()
        W1b[:, 0] = 1
        W0b = Wb.copy()
        W0b[:, 0] = 0

        return float(np.mean(mb.predict(W1b) - mb.predict(W0b)))

    se_hat, ci_95 = _bootstrap_ate(d, ate_fn, B=B, seed=seed, ci_level=ci_level)

    notes = [
        "Outcome regression plug-in estimator (NN).",
        "Same plug-in formula: mean(m_hat(1,X) - m_hat(0,X)).",
        f"Bootstrap SE/CI with B={B}.",
        f"MLP hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}, max_iter={max_iter}, early_stopping={early_stopping}."
    ]
    if status != "ok":
        notes.append(f"Overlap {status}: e_hat in [{e_min:.4g}, {e_max:.4g}].")

    return {
        "estimator_name": "ra_plugin_nn",
        "ate_hat": float(ate_hat),
        "se_hat": float(se_hat),
        "ci_95": [float(ci_95[0]), float(ci_95[1])],
        "n_used": int(n),
        "p_covariates": int(len(x_cols)),
        "overlap_status": status,
        "e_min": e_min,
        "e_max": e_max,
        "notes": " ".join(notes)
    }
