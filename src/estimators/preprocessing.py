"""Shared preprocessing for propensity-based estimators."""

import pandas as pd

def _prep_ps_rows(df: pd.DataFrame):
    if not {"Y", "A"}.issubset(df.columns):
        raise ValueError("df must contain columns 'Y' and 'A'.")

    x_cols = [c for c in df.columns if c.startswith("X")]
    d = df[["Y", "A"] + x_cols].copy()

    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d["A"] = pd.to_numeric(d["A"], errors="coerce").round()

    for c in x_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().reset_index(drop=True)
    if d.shape[0] < 5:
        raise ValueError("Not enough complete rows after dropping missing values.")

    d["A"] = d["A"].astype(int).clip(0, 1)
    if d["A"].nunique() < 2:
        raise ValueError("A must contain both 0 and 1.")

    return d, x_cols
