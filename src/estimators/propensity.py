"""Propensity-score estimation (logistic regression, random forest, or MLP)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import _prep_ps_rows

def fit_propensity_score(
    df: pd.DataFrame,
    method: str = "logit",
    random_state: int = 0,
    rf_n_estimators: int = 300,
    rf_max_depth=None,
    rf_min_samples_leaf: int = 5,
    nn_hidden_layer_sizes=(64, 32)
):
    d, x_cols = _prep_ps_rows(df)
    A = d["A"].to_numpy(dtype=int)

    if len(x_cols) == 0:
        raw_e_hat = np.full(len(d), float(A.mean()))
        method_used = "constant"
    else:
        X = d[x_cols].to_numpy(dtype=float)
        method_used = method.lower()

        if method_used == "logit":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=random_state
                ))
            ])

        elif method_used in {"rf", "random_forest"}:
            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=random_state
            )

        elif method_used in {"nn", "mlp", "neural_net", "neural_network"}:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=nn_hidden_layer_sizes
                ))
            ])

        else:
            raise ValueError("method must be one of: 'logit', 'rf', 'nn'.")

        model.fit(X, A)
        raw_e_hat = model.predict_proba(X)[:, 1].astype(float)

    return {
        "data": d,
        "x_cols": x_cols,
        "method": method_used,
        "raw_e_hat": raw_e_hat,
        "e_hat": raw_e_hat.copy(),
        "n_used": int(len(d)),
        "p_covariates": int(len(x_cols))
    }
