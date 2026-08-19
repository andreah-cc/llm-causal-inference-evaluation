"""Manual single-replicate LLM evaluation workflow.

The original capstone workflow prints a prompt, asks the researcher to run it
in ChatGPT, and then parses the pasted JSON response.
"""

import numpy as np
import pandas as pd

from .preprocessing import preclean
from .serialization import standardize_input_representation
from .prompts import FEW_SHOT_A, FEW_SHOT_B1, FEW_SHOT_B2, FEW_SHOT_C, FEW_SHOT_D, build_prompt
from .parsing import parse_llm_output_module2

# evaluation protocol and metrics
def run_single_replicate(engine_fn, engine_kwargs: dict, csv_path, protocol="A", protocol_kwargs=None): # engine_kwargs: dict or args for engine_fn e.g., (X, w, a, k, b, t, esp)
    if protocol_kwargs is None:
        protocol_kwargs = {}
    n_row_threshold = 200
    subset_n = 50
    decimals = 3
    k_smd = 10

    # generate dataset D with sample size n
    X, A, Y0, Y1, Y = engine_fn(**engine_kwargs)

    # compute ground-truth ATE
    tau_true = float(np.mean(Y1 - Y0))
    X = np.asarray(X).T
    A = np.asarray(A).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    n = X.shape[0]
    p = X.shape[1]

    df = pd.DataFrame(X, columns = [f'X{i + 1}' for i in  range(p)])
    df.insert(0, 'Y', Y)
    df.insert(0, 'A', A)

    for c in df.columns:
        if c != 'A':
            df[c] = pd.to_numeric(df[c], errors = 'coerce').round(decimals)
    df['A'] = pd.to_numeric(df['A'], errors = 'coerce').round().astype('Int64').clip(0, 1)

    df.to_csv(csv_path, index = False)
    df_clean = preclean(csv_path)
    df_clean.to_csv(csv_path, index = False)

    # ----------------------------------------------------------
    # Module 2 serialization (protocol-based)
    # ----------------------------------------------------------

    serialized_data = standardize_input_representation(
        csv_path,
        protocol= protocol,
        **protocol_kwargs
    )

    print(serialized_data)

    # query the LLM under a fixed ICL condition
    shots_map = {"A": FEW_SHOT_A, "B1": FEW_SHOT_B1, "B2": FEW_SHOT_B2, "C": FEW_SHOT_C, "D": FEW_SHOT_D}
    prompt = build_prompt(serialized_data, protocol=protocol, few_shot_examples=shots_map[protocol])
    print('\n' + 'PROMPT START' + '\n')
    print(prompt)
    print('\n' + 'PROMPT END' + '\n')
    raw_output = input('Run the prompt in ChatGPT and paste the LLM JSON output here, then press Enter:\n')

    # parse outputs and overlap warning flags
    format_failure = False
    parse_error = False
    ate_hat = L = U = overlap_warning = None
    parsed = None

    parsed_out = parse_llm_output_module2(raw_output)

    ate_hat = parsed_out["ate_hat"]
    L = parsed_out["ci95_lower"]
    U = parsed_out["ci95_upper"]
    overlap_warning = parsed_out["overlap_warning"]
    format_failure = parsed_out["format_failure"]
    parse_error = parsed_out["parse_error"]

    return{
        'csv_path': csv_path, 'n': n, 'p':p, 'tau_true': tau_true, 'ate_hat': ate_hat,
        'ci95_lower': L, 'ci95_upper': U, 'overlap_warning': overlap_warning,  'overlap_status': parsed_out['overlap_status'],
        'e_min': parsed_out['e_min'], 'e_max': parsed_out['e_max'], 'estimator_name': parsed_out['estimator_name'],
        'refusal_or_caution': parsed_out['refusal_or_caution'],'format_failure': format_failure,
        'parse_error': parse_error#, 'raw_output': raw_output, 'prompt': prompt, 'serialized_data': serialized_data,
        #'engine_kwargs': engine_kwargs
    }
