"""Protocol-specific data representations for ICL evaluation."""

import numpy as np
import pandas as pd

from src.simulation.common import sigmoid

# ----------------------------------------------------------
# PROTOCOL A — GROUP SUMMARIES
# ----------------------------------------------------------
# Converts data into group-level summaries required by
# Protocol A (Difference-in-Means).
# Outputs treated/control n, mean_Y, sd_Y.
# ----------------------------------------------------------

def standardize_protocol_A_group_summaries(df):
    treated = df[df["A"] == 1]["Y"].dropna()
    control = df[df["A"] == 0]["Y"].dropna()

    n1 = int(treated.shape[0])
    n0 = int(control.shape[0])

    mean1 = float(treated.mean()) if n1 > 0 else None
    mean0 = float(control.mean()) if n0 > 0 else None
    sd1 = float(treated.std(ddof=1)) if n1 > 1 else None
    sd0 = float(control.std(ddof=1)) if n0 > 1 else None

    lines = []
    lines.append("GROUP_SUMMARIES_START")
    lines.append("group n mean_Y sd_Y")
    lines.append(f"treated {n1} {mean1} {sd1}")
    lines.append(f"control {n0} {mean0} {sd0}")
    lines.append("GROUP_SUMMARIES_END")

    return "\n".join(lines)
# ----------------------------------------------------------
# PROTOCOL B1 — MICRODATA (REGRESSION ADJUSTMENT)
# ----------------------------------------------------------
# Provides a small microdata table with columns:
#   A Y X1 X2 ... Xp
# Used when n is small enough to fit in the prompt.
# ----------------------------------------------------------

def standardize_protocol_B_microdata(df, max_rows=200, seed=0):
    import numpy as np
    import pandas as pd

    x_cols = [c for c in df.columns if c.startswith("X")]
    keep = [c for c in (["A", "Y"] + x_cols) if c in df.columns]
    tmp = df[keep].copy()

    n = tmp.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        tmp = tmp.iloc[idx].reset_index(drop=True)
        sample_note = f"subsampled {max_rows} of {n} rows (seed={seed})"
    else:
        sample_note = "no_rows_omitted"

    lines = []
    lines.append("MICRODATA_START")
    lines.append(f"note {sample_note} missing_values=None")
    lines.append(" ".join(keep))

    for _, row in tmp.iterrows():
        vals = []
        for c in keep:
            v = row[c]
            if pd.isna(v):
                vals.append("None")
            else:
                if c == "A":
                    vals.append(str(int(round(float(v)))))
                else:
                    vals.append(str(float(v)))
        lines.append(" ".join(vals))

    lines.append("MICRODATA_END")
    return "\n".join(lines)
# ----------------------------------------------------------
# PROTOCOL B2 — COMPRESSED REGRESSION OUTPUT
# ----------------------------------------------------------
# Computes regression output externally and serializes:
#   coef_A and (robust) se_A for Y ~ A + X (+ intercept)
# LLM should report implied ATE = coef_A (and CI if instructed).
# ----------------------------------------------------------

def standardize_protocol_B_regression_output(
    df,
    max_rows=200,
    seed=0,
    mice_seed: int = 0,
    mice_max_iter: int = 10
):
    import numpy as np
    import pandas as pd

    # Select covariates X* and keep consistent column order with Protocol B1
    x_cols = [c for c in df.columns if c.startswith("X")]
    keep = [c for c in (["A", "Y"] + x_cols) if c in df.columns]
    tmp = df[keep].copy()

    # Subsample first (same step/order as Protocol B1), then run regression on this micro-sample
    n0 = tmp.shape[0]
    if n0 > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n0, size=max_rows, replace=False)
        tmp = tmp.iloc[idx].reset_index(drop=True)
        sample_note = f"subsampled {max_rows} of {n0} rows (seed={seed})"
    else:
        sample_note = "no_rows_omitted"

    # Regression requires numeric A/Y (drop rows with missing A or Y)
    tmp["A"] = pd.to_numeric(tmp["A"], errors="coerce")
    tmp["Y"] = pd.to_numeric(tmp["Y"], errors="coerce")
    tmp = tmp.dropna(subset=["A", "Y"]).copy()

    # Force A into {0,1} (mirrors how A is serialized in Protocol B1)
    tmp["A"] = tmp["A"].round().clip(0, 1)

    A = tmp["A"].astype(float).to_numpy()
    Y = tmp["Y"].astype(float).to_numpy()
    n = int(len(tmp))

    # Impute missing X using MICE (IterativeImputer) so the regression can run
    x_cols_present = [c for c in x_cols if c in tmp.columns]
    if len(x_cols_present) > 0:
        for c in x_cols_present:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        X_raw = tmp[x_cols_present].to_numpy(dtype=float)

        # Enable IterativeImputer (MICE-style) in sklearn
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=int(mice_max_iter),
            random_state=int(mice_seed),
            sample_posterior=False,      # deterministic given seed
            initial_strategy="mean"
        )
        X = imp.fit_transform(X_raw).astype(float)
        impute_note = f"MICE(IterativeImputer,BayesianRidge,max_iter={mice_max_iter},seed={mice_seed})"
    else:
        X = np.zeros((n, 0), dtype=float)
        impute_note = "none(no X)"

    # Design matrix: intercept + A + X
    Z = np.column_stack([np.ones(n), A, X])

    coef_A = None
    se_A = None
    se_type = "None"
    r2 = None

    try:
        import statsmodels.api as sm
        # HC1 robust SE for coef on A
        res = sm.OLS(Y, Z).fit(cov_type="HC1")
        coef_A = float(res.params[1])
        se_A = float(res.bse[1])
        se_type = "HC1"
        r2 = float(res.rsquared)
    except Exception:
        # Fallback: classic OLS SE computed manually
        XtX_inv = np.linalg.pinv(Z.T @ Z)
        beta = XtX_inv @ (Z.T @ Y)
        resid = Y - Z @ beta
        dof = max(n - Z.shape[1], 1)
        s2 = float((resid @ resid) / dof)
        var = s2 * XtX_inv
        coef_A = float(beta[1])
        se_A = float(np.sqrt(np.diag(var))[1])
        se_type = "classic"
        tss = float(np.sum((Y - Y.mean()) ** 2))
        rss = float(np.sum(resid ** 2))
        r2 = float(1 - rss / tss) if tss > 0 else None

    # Serialize compact regression output for Protocol B2
    lines = []
    lines.append("REGRESSION_OUTPUT_START")
    lines.append("model Y ~ A + X intercept 1")
    lines.append(f"note {sample_note}")
    lines.append(f"x_imputation {impute_note}")
    lines.append(f"n {n} p_X {int(len(x_cols_present))} r2 {r2}")
    lines.append(f"se_type {se_type}")
    lines.append("coef_A se_A")
    lines.append(f"{coef_A} {se_A}")
    lines.append("REGRESSION_OUTPUT_END")
    return "\n".join(lines)

# ----------------------------------------------------------
# PROTOCOL C — PROPENSITY SCORE BIN SUMMARIES
# ----------------------------------------------------------
# Creates K quantile bins of a simplified propensity score
# and reports bin-level summaries:
#   w_k, n_treated, n_control,
#   meanY_treated, meanY_control
# Used for PS stratification estimation.
# ----------------------------------------------------------


def standardize_protocol_C_ps_bin_summaries(df, K=5):
    x_cols = [c for c in df.columns if c.startswith("X")]
    n = df.shape[0]

    # Simple propensity score approximation
    if len(x_cols) == 0:
        e_hat = np.full(n, float(df["A"].mean()))
        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
    else:
        X = df[x_cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0) # replace missing values with 0
        A = df["A"].to_numpy().astype(float)
        A_center = A - A.mean()
        beta = X.T @ A_center # beta measures how strongly feature correlates with treatment
        norm = np.linalg.norm(beta)
        if norm > 0:
            beta = beta / norm
        score = X @ beta # every row recieves score on how aligned with treatment direction
        e_hat = sigmoid(score) #[0,1]
        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

    tmp = df.copy()
    tmp["e_hat"] = e_hat

    # Quantile binning
    qs = np.linspace(0, 1, K + 1)
    edges = np.unique(np.quantile(e_hat, qs))

    if len(edges) < 3:
        tmp["bin"] = 0
        K_eff = 1
    else:
        tmp["bin"] = pd.cut(tmp["e_hat"], bins=edges, include_lowest=True, labels=False)
        K_eff = int(tmp["bin"].nunique())

    lines = []
    lines.append("PS_BIN_SUMMARIES_START")
    lines.append("bin w_k n_treated n_control meanY_treated meanY_control")

    for b in range(K_eff):
        d = tmp[tmp["bin"] == b]
        w_k = float(len(d) / len(tmp))

        y1 = d[d["A"] == 1]["Y"].dropna()
        y0 = d[d["A"] == 0]["Y"].dropna()

        n1 = int(y1.shape[0])
        n0 = int(y0.shape[0])

        mean1 = float(y1.mean()) if n1 > 0 else None
        mean0 = float(y0.mean()) if n0 > 0 else None

        lines.append(f"{b} {w_k} {n1} {n0} {mean1} {mean0}")

    lines.append("PS_BIN_SUMMARIES_END")
    return "\n".join(lines)


# ----------------------------------------------------------
# PROTOCOL D — AIPW TABLE
# ----------------------------------------------------------
# Converts data into per-unit nuisance quantities required by
# Protocol D (Doubly Robust AIPW).
# Outputs A, Y, e_hat, mu0_hat, mu1_hat.
# All nuisance components are computed externally here.
# ----------------------------------------------------------

def standardize_protocol_D_aipw_table(df, max_rows=200, seed=0):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, LinearRegression

    x_cols = [c for c in df.columns if c.startswith("X")]

    A = df["A"].astype(int).to_numpy()
    Y = df["Y"].to_numpy(dtype=float)
    n = len(A)

    # -------------------------
    # Propensity score (logistic)
    # -------------------------
    if len(x_cols) == 0:
        e_hat = np.full(n, float(np.mean(A)))
        e_hat = np.clip(e_hat, 0.01, 0.99)
    else:
        X = df[x_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        ps_model = LogisticRegression(solver="liblinear", max_iter=1000)
        ps_model.fit(X, A)
        e_hat = ps_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 0.01, 0.99)
    # -------------------------
    # Outcome models
    # -------------------------
    obs = pd.notna(df["Y"]).to_numpy()

    if len(x_cols) == 0:
        mu1 = float(np.nanmean(Y[(A == 1) & obs])) if np.any((A == 1) & obs) else np.nan
        mu0 = float(np.nanmean(Y[(A == 0) & obs])) if np.any((A == 0) & obs) else np.nan
        mu1_hat = np.full(n, mu1)
        mu0_hat = np.full(n, mu0)
    else:
        X = df[x_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        X_obs = X[obs]
        Y_obs = Y[obs]
        A_obs = A[obs]

        global_mean = float(np.mean(Y_obs)) if len(Y_obs) else np.nan
        mu0_hat = np.full(n, global_mean)
        mu1_hat = np.full(n, global_mean)

        if np.any(A_obs == 0):
            m0 = LinearRegression()
            m0.fit(X_obs[A_obs == 0], Y_obs[A_obs == 0])
            mu0_hat = m0.predict(X)

        if np.any(A_obs == 1):
            m1 = LinearRegression()
            m1.fit(X_obs[A_obs == 1], Y_obs[A_obs == 1])
            mu1_hat = m1.predict(X)

    # -------------------------
    # OUTPUT: per-unit AIPW table (matches Protocol D prompt)
    # -------------------------
    keep = np.where(obs)[0]
    if keep.size > max_rows:
        rng = np.random.default_rng(seed)
        keep = keep[:max_rows]

    lines = []
    lines.append("AIPW_TABLE_START")
    lines.append("A Y e_hat mu0_hat mu1_hat") # fit model on small data, get fitted function, predict on big data
    for i in keep:
        lines.append(
            f"{int(A[i])} {float(Y[i]):.6f} {float(e_hat[i]):.6f} "
            f"{float(mu0_hat[i]):.6f} {float(mu1_hat[i]):.6f}"
        )
    lines.append("AIPW_TABLE_END")
    return "\n".join(lines)

# ----------------------------------------------------------
# UNIFIED INTERFACE
# ----------------------------------------------------------
# Main entry point for Module 2 input standardization.
# Converts cleaned CSV data into protocol-specific text.
# ----------------------------------------------------------

def standardize_input_representation(csv_path, protocol="A", **kwargs):
    protocol = protocol.upper().strip()
    df = preclean(csv_path)

    if protocol == "A":
        return standardize_protocol_A_group_summaries(df)
    elif protocol == "B2":
        return standardize_protocol_B_regression_output(df)
    elif protocol == "B1":
        max_rows = kwargs.get("max_rows", 200)
        seed = kwargs.get("seed", 0)
        return standardize_protocol_B_microdata(df, max_rows=max_rows, seed=seed)
    elif protocol == "C":
      K = kwargs.get("K", 5)
      return standardize_protocol_C_ps_bin_summaries(df, K=K)
    elif protocol == "D":
      max_rows = kwargs.get("max_rows", 200)
      seed = kwargs.get("seed", 0)
      return standardize_protocol_D_aipw_table(df, max_rows=max_rows, seed=seed)
    else:
        raise ValueError("protocol must be one of: A, B1, B2, C, D")
