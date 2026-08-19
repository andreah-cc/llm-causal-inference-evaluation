"""Pre-clean simulated datasets before LLM serialization."""

import numpy as np
import pandas as pd

# precleaning of the data
def preclean(csv_path, decimals=3):
    df = pd.read_csv(csv_path)

    x_prefix = "X"
    x_cols = [c for c in df.columns if c.startswith(x_prefix)]

    # Sort X columns as X1, X2, ..., Xp
    def x_key(name: str):
        suffix = name[len(x_prefix):]
        return int(suffix) if suffix.isdigit() else suffix
    x_cols = sorted(x_cols, key=x_key)

    # Keep only required columns
    keep = [c for c in (["A", "Y"] + x_cols) if c in df.columns]
    df = df[keep].copy()

    # Convert to numeric and handle infinities
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Enforce binary treatment
    if "A" in df.columns:
        df["A"] = df["A"].round().astype("Int64").clip(0, 1)

    # Standardize numeric precision
    for c in df.columns:
        if c != "A":
            df[c] = df[c].round(decimals)

    return df
