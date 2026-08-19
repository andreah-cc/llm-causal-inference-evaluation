"""Utilities for serializing simulated data for LLM evaluation."""

import json
import numpy as np
import pandas as pd


def small_to_text(csv_path):
    df = pd.read_csv(csv_path)
    lines = ["DATA_START", " ".join(df.columns)]
    lines.extend(" ".join(map(str, row.values)) for _, row in df.iterrows())
    lines.append("DATA_END")
    return "\n".join(lines)


def large_to_subset(csv_path, subset_n=50, random_state=0):
    df = pd.read_csv(csv_path)
    half = subset_n // 2
    treated = df[df["A"] == 1].sample(n=half, replace=False, random_state=random_state)
    untreated = df[df["A"] == 0].sample(n=half, replace=False, random_state=random_state)
    subset = pd.concat([treated, untreated], axis=0)

    lines = ["SUBSETTED_DATA_START", " ".join(df.columns)]
    lines.extend(" ".join(map(str, row.values)) for _, row in subset.iterrows())
    lines.append("SUBSETTED_DATA_END")
    return "\n".join(lines)


def summary_stats(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    x_cols = [c for c in df.columns if c.startswith("X")]
    for col in x_cols:
        treated = df[df["A"] == 1][col].dropna()
        control = df[df["A"] == 0][col].dropna()
        if len(treated) == 0 or len(control) == 0:
            continue

        mean_A1 = treated.mean()
        mean_A0 = control.mean()
        sd_A1 = treated.std(ddof=1)
        sd_A0 = control.std(ddof=1)
        pooled_sd = np.sqrt(0.5 * (sd_A1**2 + sd_A0**2))
        smd = 0.0 if pooled_sd == 0 else (mean_A1 - mean_A0) / pooled_sd
        results.append((col, mean_A0, mean_A1, sd_A0, sd_A1, smd))

    return results


def preclean(csv_path, decimals=3, standardize_x=False, row_shuffle_seed=None):
    df = pd.read_csv(csv_path)
    x_cols = [c for c in df.columns if c.startswith("X")]

    def x_key(name):
        suffix = name[1:]
        return int(suffix) if suffix.isdigit() else suffix

    x_cols = sorted(x_cols, key=x_key)
    keep = [c for c in (["A", "Y"] + x_cols) if c in df.columns]
    df = df[keep]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    if row_shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=row_shuffle_seed).reset_index(drop=True)

    if "A" in df.columns:
        df["A"] = df["A"].round().astype("Int64").clip(lower=0, upper=1)

    if standardize_x and x_cols:
        x_present = [c for c in x_cols if c in df.columns]
        means = df[x_present].mean(skipna=True)
        stds = df[x_present].std(skipna=True, ddof=0).replace(0, np.nan)
        df[x_present] = ((df[x_present] - means) / stds).fillna(0.0)

    for c in df.columns:
        if c != "A":
            df[c] = df[c].round(decimals)

    return df


def build_prompt(serialized_data):
    instructions = [
        "TASK_INSTRUCTIONS",
        "You are given observational data with covariates X, treatment A (0/1), and outcome Y.",
        "Estimate the average treatment effect (ATE) from observational data (X, A, Y).",
        "Assume unconfoundedness given X and discuss overlap/positivity.",
        "Provide a point estimate and a 95% interval, with warnings if overlap is weak.",
        "END_TASK_INSTRUCTIONS",
    ]

    schema = [
        "OUTPUT_SCHEMA",
        "Return exactly ONE JSON object with these required fields:",
        "{",
        '  "ate_hat": <number>,',
        '  "ci95": [<lower>, <upper>],',
        '  "method": "<short label>",',
        '  "assumptions": ["unconfoundedness_given_X", "overlap"],',
        '  "overlap_warning": <true/false>,',
        '  "notes": "<brief justification, <= 120 words>"',
        "}",
        "END_OUTPUT_SCHEMA",
    ]

    return "\n".join(instructions + ["", serialized_data.strip(), ""] + schema)
