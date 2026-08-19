"""Generate Module 4 prompts from simulation engines.

The uploaded Module 4 notebook implements the 500-simulation grid only for
Engine 1 and explicitly leaves Engines 2-8 as a TODO. This module preserves
that scope rather than inventing missing experiment settings.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.llm.preprocessing import preclean
from src.llm.serialization import standardize_input_representation
from src.llm.prompts import (
    FEW_SHOT_A,
    FEW_SHOT_B1,
    FEW_SHOT_B2,
    FEW_SHOT_C,
    FEW_SHOT_D,
    build_prompt,
)
from src.simulation.engine1 import main_1


SHOTS_MAP = {
    "A": FEW_SHOT_A,
    "B1": FEW_SHOT_B1,
    "B2": FEW_SHOT_B2,
    "C": FEW_SHOT_C,
    "D": FEW_SHOT_D,
}


def build_replicate_prompt(
    engine_fn,
    engine_kwargs: dict,
    csv_path,
    protocol="A",
    protocol_kwargs=None,
    decimals=3,
):
    """Generate one dataset and return its final ICL prompt."""
    if protocol_kwargs is None:
        protocol_kwargs = {}

    X, A, Y0, Y1, Y = engine_fn(**engine_kwargs)

    X = np.asarray(X).T
    A = np.asarray(A).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    n, p = X.shape
    df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
    df.insert(0, "Y", Y)
    df.insert(0, "A", A)

    for c in df.columns:
        if c != "A":
            df[c] = pd.to_numeric(df[c], errors="coerce").round(decimals)
    df["A"] = (
        pd.to_numeric(df["A"], errors="coerce")
        .round()
        .astype("Int64")
        .clip(0, 1)
    )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    df_clean = preclean(csv_path)
    df_clean.to_csv(csv_path, index=False)

    serialized_data = standardize_input_representation(
        csv_path,
        protocol=protocol,
        **protocol_kwargs,
    )

    prompt = build_prompt(
        serialized_data,
        protocol=protocol,
        few_shot_examples=SHOTS_MAP[protocol],
    )

    return "PROMPT START\n" + prompt + "\nPROMPT END"


def build_engine1_grid(
    X,
    w,
    b,
    eps,
    reps=10,
    taus=(0, 0.5, 1),
    protocol="D",
    work_dir="tmp/module4",
):
    """Build the Engine 1 simulation grid implemented in the source notebook."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    final = []

    for tau in taus:
        for i in range(reps):
            csv_path = work_dir / f"engine1_tau_{tau}_rep_{i}.csv"
            base_prompt = build_replicate_prompt(
                main_1,
                engine_kwargs={
                    "X": X,
                    "w": w,
                    "a": 1,
                    "k": 1,
                    "b": b,
                    "tau": tau,
                    "eps": eps,
                },
                csv_path=csv_path,
                protocol=protocol,
            )

            prompt_id = f"tau_{tau}_rep_{i}"
            final.append(
                {
                    "engine": 1,
                    "tau": tau,
                    "prompt_id": prompt_id,
                    "base_prompt": base_prompt,
                }
            )

    return final
