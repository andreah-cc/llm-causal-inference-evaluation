"""Few-shot examples and structured prompt construction for Protocols A-D."""

import json

# few-shot
FEW_SHOT_A = [
  # 1) Clean overlap and simple numbers
  {
    "dataset": """
GROUP_SUMMARIES_START
group n mean_Y sd_Y
treated 100 10.0 2.0
control 120 9.0 2.5
GROUP_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": 1.0,
      "se_hat": 0.3034523576,
      "ci_95": [0.4052333791, 1.5947666209],
      "estimator_name": "diff_in_means",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity information provided in GROUP_SUMMARIES; overlap not assessable."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided",
        "model_specification_risk": "high"
      },
      "refusal_or_caution": "Unadjusted difference-in-means from group summaries; may be biased under confounding (no covariate adjustment shown)."
    }
  },

  # 2) Missing treated SD
  {
    "dataset": """
GROUP_SUMMARIES_START
group n mean_Y sd_Y
treated 1 5.0 None
control 90 4.0 1.5
GROUP_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": 1.0,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "diff_in_means",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity information provided; overlap not assessable."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "sd_Y missing (treated sd_Y=None because n_treated<=1) -> cannot compute se_hat or ci_95",
        "model_specification_risk": "high"
      },
      "refusal_or_caution": "Computed ate_hat from mean_Y values, but sd_Y inputs are incomplete; se_hat and ci_95 are null per contract."
    }
  },

  # 3) Missing required inputs(No control observations)
  {
    "dataset": """
GROUP_SUMMARIES_START
group n mean_Y sd_Y
treated 50 2.0 1.0
control 0 None None
GROUP_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": None,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "diff_in_means",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity information provided; overlap not assessable."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "control group has n=0 -> mean_Y_control=None -> cannot compute ate_hat",
        "model_specification_risk": "high"
      },
      "refusal_or_caution": "Cannot compute difference-in-means ATE because control mean_Y is missing (n_control=0)."
    }
  }
]

FEW_SHOT_B1 = [
  # 1) clean microdata
  {
    "dataset": """
MICRODATA_START
note no_rows_omitted missing_values=None
A Y X1
0 10.0 0.0
0 13.0 1.0
1 12.0 0.0
1 15.0 1.0
MICRODATA_END
""".strip(),
    "answer_json": {
      "ate_hat": 2.0,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ra_ols",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity score information provided in microdata-only Protocol B1."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided (dropped_rows_YA=0, imputed_X_rows=0)",
        "model_specification_risk": "depends_on_linear_model_fit"
      },
      "refusal_or_caution": "ate_hat computed as the OLS coefficient on A from Y ~ A + X with intercept; SE/CI omitted because no variance recipe was provided."
    }
  },

  # 2) Missing X value
  {
    "dataset": """
MICRODATA_START
note no_rows_omitted missing_values=None
A Y X1
0 10.0 0.0
0 13.0 None
1 12.0 0.0
1 15.0 1.0
MICRODATA_END
""".strip(),
    "answer_json": {
      "ate_hat": 2.0,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ra_ols",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity score information provided in Protocol B1."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "Imputed X1=0.0 for 1 row where X1=None; dropped_rows_YA=0",
        "model_specification_risk": "depends_on_linear_model_fit"
      },
      "refusal_or_caution": "Computed ate_hat after deterministic X-imputation (None -> 0.0). SE/CI omitted without an explicit variance recipe."
    }
  },

  # 3) Missing Y value
  {
    "dataset": """
MICRODATA_START
note no_rows_omitted missing_values=None
A Y X1
0 None 0.0
1 12.0 0.0
MICRODATA_END
""".strip(),
    "answer_json": {
      "ate_hat": None,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ra_ols",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity score information provided."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "Dropped 1 row with Y=None; remaining usable rows < 3 -> cannot fit OLS reliably",
        "model_specification_risk": "unknown"
      },
      "refusal_or_caution": "Cannot compute OLS regression-adjusted ATE because there are too few usable rows after dropping missing Y/A."
    }
  }
]

FEW_SHOT_B2 = [
  # 1) Robust regression output
  {
    "dataset": """
REGRESSION_OUTPUT_START
model Y ~ A + X intercept 1
n 200 p_X 5 r2 0.42
se_type HC1
coef_A se_A
1.2 0.5
REGRESSION_OUTPUT_END
""".strip(),
    "answer_json": {
      "ate_hat": 1.2,
      "se_hat": 0.5,
      "ci_95": [0.22, 2.18],
      "estimator_name": "ra_ols",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity score information provided in Protocol B2."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided",
        "model_specification_risk": "depends_on_linear_model_fit"
      },
      "refusal_or_caution": "Used provided regression output: ate_hat=coef_A, se_hat=se_A, CI computed as coef_A ± 1.96*se_A (se_type=HC1)."
    }
  },

  # 2) Classic SE regression output
  {
    "dataset": """
REGRESSION_OUTPUT_START
model Y ~ A + X intercept 1
n 50 p_X 1 r2 0.10
se_type classic
coef_A se_A
-0.8 0.4
REGRESSION_OUTPUT_END
""".strip(),
    "answer_json": {
      "ate_hat": -0.8,
      "se_hat": 0.4,
      "ci_95": [-1.584, -0.016],
      "estimator_name": "ra_ols",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "unknown",
        "e_min": None,
        "e_max": None,
        "notes": "No propensity score information provided in Protocol B2."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided",
        "model_specification_risk": "depends_on_linear_model_fit"
      },
      "refusal_or_caution": "Used provided regression output; CI computed as coef_A ± 1.96*se_A (se_type=classic)."
    }
  }
]

FEW_SHOT_C = [
  # 1) Clean overlap
  {
    "dataset": """
PS_BIN_SUMMARIES_START
bin w_k n_treated n_control meanY_treated meanY_control
0 0.5 50 50 3.0 2.0
1 0.5 50 50 4.0 2.0
PS_BIN_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": 1.5,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ps_strat",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "ok",
        "e_min": None,
        "e_max": None,
        "notes": "All bins have n_treated>0 and n_control>0; no empty cells. (Propensity values not provided, so e_min/e_max are null.)"
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided",
        "model_specification_risk": "medium"
      },
      "refusal_or_caution": "ATE computed via weighted within-bin contrasts; SE/CI omitted because no bin-level variance recipe was provided."
    }
  },

  # 2) Explicit overlap violation(empty treated cell in a bin)
  {
    "dataset": """
PS_BIN_SUMMARIES_START
bin w_k n_treated n_control meanY_treated meanY_control
0 0.4 30 30 5.0 4.0
1 0.3 0 40 None 3.0
2 0.3 20 20 4.0 3.5
PS_BIN_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": None,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ps_strat",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "violation",
        "e_min": None,
        "e_max": None,
        "notes": "Overlap violation: at least one bin has an empty treatment cell (bin 1: n_treated=0)."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "bin 1 has meanY_treated=None due to n_treated=0 -> cannot compute weighted ATE",
        "model_specification_risk": "high"
      },
      "refusal_or_caution": "Cannot compute PS-stratified ATE because positivity is violated (empty treated cell) and required bin mean is missing; set ate_hat=null and CI null."
    }
  },

  # 3) Explicit overlap violation(empty control cell in a bin)
  {
    "dataset": """
PS_BIN_SUMMARIES_START
bin w_k n_treated n_control meanY_treated meanY_control
0 0.6 40 0 2.0 None
1 0.4 40 80 3.0 2.0
PS_BIN_SUMMARIES_END
""".strip(),
    "answer_json": {
      "ate_hat": None,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "ps_strat",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "violation",
        "e_min": None,
        "e_max": None,
        "notes": "Overlap violation: at least one bin has an empty control cell (bin 0: n_control=0)."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "bin 0 has meanY_control=None due to n_control=0 -> cannot compute weighted ATE",
        "model_specification_risk": "high"
      },
      "refusal_or_caution": "Cannot compute PS-stratified ATE because positivity is violated (empty control cell) and required bin mean is missing; set ate_hat=null and CI null."
    }
  }
]

FEW_SHOT_D = [
  # 1) Clean Overlap: non-extreme e_hat
  {
    "dataset": """
AIPW_TABLE_START
A Y e_hat mu0_hat mu1_hat
1 3.000 0.500 1.500 2.500
0 1.000 0.500 1.000 2.000
AIPW_TABLE_END
""".strip(),
    "answer_json": {
      "ate_hat": 1.5,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "aipw_dr",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "ok",
        "e_min": 0.5,
        "e_max": 0.5,
        "notes": "All e_hat are moderate (not near 0 or 1)."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided (dropped_rows=0, valid_rows=2)",
        "model_specification_risk": "depends_on_nuisance_quality"
      },
      "refusal_or_caution": "Point estimate computed via AIPW psi_i formula from provided nuisance outputs; SE/CI omitted because no variance recipe was provided."
    }
  },

  # 2) Explicit overlap violation: e_hat extremely close to 0/1, but ATE is still computable
  {
    "dataset": """
AIPW_TABLE_START
A Y e_hat mu0_hat mu1_hat
1 2.000 0.990 1.200 1.800
0 1.000 0.020 0.900 1.400
1 3.000 0.800 1.500 2.600
AIPW_TABLE_END
""".strip(),
    "answer_json": {
      "ate_hat": 0.9333,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "aipw_dr",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "warning",
        "e_min": 0.02,
        "e_max": 0.99,
        "notes": "Weak overlap: some e_hat values are extremely close to 0 or 1 (e_max>=0.99). AIPW weights can be unstable."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "not_provided (dropped_rows=0, valid_rows=3)",
        "model_specification_risk": "high_due_to_extreme_weights"
      },
      "refusal_or_caution": "ATE point estimate is computable, but inference is likely unstable under weak overlap (extreme propensity scores). Avoid reporting a precise CI; consider trimming/truncation or improved nuisance modeling."
    }
  },

  # 3) Missing required inputs: mu0_hat/mu1_hat missing
  {
    "dataset": """
AIPW_TABLE_START
A Y e_hat mu0_hat mu1_hat
1 2.000 0.600 null 1.900
0 1.000 0.400 0.800 null
AIPW_TABLE_END
""".strip(),
    "answer_json": {
      "ate_hat": None,
      "se_hat": None,
      "ci_95": [None, None],
      "estimator_name": "aipw_dr",
      "assumptions": ["ignorability", "overlap", "SUTVA"],
      "overlap_assessment": {
        "status": "ok",
        "e_min": 0.4,
        "e_max": 0.6,
        "notes": "Propensity scores are moderate, but key nuisance predictions are missing so psi_i cannot be computed."
      },
      "diagnostics": {
        "balance_summary": "not_provided",
        "missingness": "Dropped rows with missing/non-numeric mu0_hat or mu1_hat (dropped_rows=2, valid_rows=0) -> cannot compute ate_hat.",
        "model_specification_risk": "unknown"
      },
      "refusal_or_caution": "Cannot compute AIPW ATE because required nuisance predictions (mu0_hat/mu1_hat) are missing, leaving zero valid rows."
    }
  }
]

# make output machine-gradable

MODULE2_OUTPUT_SCHEMA = {
  "ate_hat": "number|null",
  "se_hat": "number|null",
  "ci_95": ["number|null", "number|null"],
  "estimator_name": "string",
  "assumptions": ["string"],
  "overlap_assessment": {
    "status": "ok|warning|violation|unknown",
    "e_min": "number|null",
    "e_max": "number|null",
    "notes": "string"
  },
  "diagnostics": {
    "balance_summary": "string|null",
    "missingness": "string|null",
    "model_specification_risk": "string|null"
  },
  "refusal_or_caution": "string|null"
}

def build_prompt(serialized_data,
                 protocol="A",
                 few_shot_examples=None,
                 include_schema_reminder=True):

    protocol = protocol.upper().strip()

    # protocol-specific instructions
    if protocol == "A":
        proto = [
            "PROTOCOL_A: Difference-in-Means from GROUP_SUMMARIES",
            "You will be given treated and control group summaries: n, mean_Y, sd_Y.",

            # Input format alignment
            "Input format:",
            "  - The EVALUATION_DATASET block contains group summaries (and may also include an optional balance/SMD table).",
            "  - Expect lines like:",
            "      T=1: n1=..., meanY1=..., sdY1=...",
            "      T=0: n0=..., meanY0=..., sdY0=...",
            "  - Treat any of these as missing tokens if they appear: null / None / NA.",

            # formula
            "Compute:",
            "  ate_hat = meanY1 - meanY0",
            "  se_hat = sqrt(sdY1^2/n1 + sdY0^2/n0)",
            "  ci_95  = [ate_hat - 1.96*se_hat, ate_hat + 1.96*se_hat]",

            # Missingness handling
            "Rules (STRICT):",
            "  - If any required quantity for SE/CI is missing (n1, n0, sdY1, sdY0), set se_hat=null and ci_95=[null, null].",
            "  - If meanY1 or meanY0 is missing, set ate_hat=null (and consequently se_hat=null and ci_95=[null, null]).",
            "  - Overlap: if no propensity information is provided in the dataset, set overlap_assessment.status='unknown', e_min=null, e_max=null, and notes explaining why."
        ]

    elif protocol == "B1":
        proto = [
            "PROTOCOL_B1: Microdata Regression Adjustment (OLS)",
            "You will be given a microdata table delimited by:",
            "  MICRODATA_START ... MICRODATA_END",
            "Format (STRICT):",
            "  - First line contains a note (ignore it except for missing token info).",
            "  - Next line is the header with columns: A Y X1 X2 ... (some X may be absent).",
            "  - Each subsequent row is whitespace-separated values in the same order.",
            "  - Missing values appear as the literal token: None (treat 'None' as missing).",

            "Goal:",
            "  - Estimate ATE using regression adjustment by fitting OLS with intercept:",
            "      Y ~ A + X (include all X columns provided).",
            "  - ate_hat = coefficient on A.",

            "Missingness rules (STRICT):",
            "  - Drop rows with missing Y or missing A.",
            "  - For missing X values (token None), impute X=0.0 (deterministic).",
            "  - If after dropping missing Y/A there are fewer than 3 usable rows, set ate_hat=null.",

            "Uncertainty:",
            "  - Unless an explicit variance/SE recipe is provided, set se_hat=null and ci_95=[null, null].",
            "  - Do NOT invent bootstrap or regression SEs.",

            "Overlap:",
            "  - No propensity information is provided here, set overlap_assessment.status='unknown', e_min=null, e_max=null.",
        ]

    elif protocol == "B2":
        proto = [
            "PROTOCOL_B2: Compressed Regression Output (Regression Adjustment)",
            "You will be given regression output delimited by:",
            "  REGRESSION_OUTPUT_START ... REGRESSION_OUTPUT_END",
            "Format (STRICT):",
            "  - Read n, p_X, r2 (may be None), se_type.",
            "  - The final 2-number row under 'coef_A se_A' contains coef_A and se_A.",
            "Computation:",
            "  - ate_hat = coef_A",
            "  - se_hat  = se_A",
            "  - ci_95   = [ate_hat - 1.96*se_hat, ate_hat + 1.96*se_hat]",
            "Rules:",
            "  - Do NOT recompute regression from scratch.",
            "  - If coef_A or se_A is missing/non-numeric, set ate_hat/se_hat null and ci_95=[null, null].",
            "Overlap:",
            "  - No propensity information is provided here, set overlap_assessment.status='unknown', e_min=null, e_max=null.",
        ]

    elif protocol == "C":
        proto = [
            "PROTOCOL_C: Propensity Score Stratification from AIPW_BIN_SUMMARIES",
            "You will be given K bins with: w_k, n_treated, n_control, meanY_treated, meanY_control (optional within-bin SDs).",

            # Input format alignment
            "Input format (STRICT):",
            "  - The EVALUATION_DATASET block contains bin summaries (and may include a line stating K).",
            "  - Each bin provides: w_k, n1k, n0k, meanY1k, meanY0k (optionally sdY1k, sdY0k).",
            "  - Treat any of these as missing tokens if they appear: null / None / NA.",

            # formula
            "Compute:",
            "  ate_hat = sum_k w_k * (meanY1k - meanY0k)",

            # Overlap assessment
            "Overlap / positivity (STRICT):",
            "  - If ANY bin has n1k==0 OR n0k==0, set overlap_assessment.status='violation' and explain in overlap_assessment.notes.",
            "  - In violation, set se_hat=null and ci_95=[null, null].",
            "  - You may still report ate_hat if it is computable from the provided (non-missing) bin means; otherwise set ate_hat=null.",

            # Missingness handling
            "Missingness (STRICT):",
            "  - If any bin needed for the sum has missing/non-numeric w_k or meanY values, set ate_hat=null and explain in diagnostics.missingness.",

            # Uncertainty/CI
            "Uncertainty:",
            "  - Unless within-bin SDs AND an explicit variance recipe are provided, set se_hat=null and ci_95=[null, null] (IMPORTANT: ci_95 must always be a 2-element list).",

            "Notes about e_min/e_max:",
            "  - If propensities themselves are not provided (only bins), set overlap_assessment.e_min=null and e_max=null."
        ]

    elif protocol == "D":
        proto = [
            "PROTOCOL_D: AIPW_TABLE",

            # Input format alignment
            "Input format:",
            "  - The EVALUATION_DATASET block will contain an AIPW table delimited by:",
            "      AIPW_TABLE_START ... AIPW_TABLE_END",
            "  - The header row is exactly:",
            "      A Y e_hat mu0_hat mu1_hat",
            "  - Each subsequent row contains 5 whitespace-separated values in that order.",
            "  - Missing values may appear as: null / None / NA (treat any of these as missing).",

            # formula
            "Compute for each valid row i (all required fields present and finite):",
            "  psi_i = (mu1_hat_i - mu0_hat_i) + A_i*(Y_i - mu1_hat_i)/e_hat_i - (1-A_i)*(Y_i - mu0_hat_i)/(1-e_hat_i)",
            "Then:",
            "  ate_hat = mean_i(psi_i) over valid rows only.",

            # Missingness handling
            "Missingness rules:",
            "  - If any of {Y, e_hat, mu0_hat, mu1_hat} is missing/non-numeric in a row, drop that row from psi_i computation.",
            "  - If zero valid rows remain, set ate_hat=null and explain in diagnostics.missingness.",
            "  - Always report in diagnostics.missingness how many rows were dropped (if any).",

            # Overlap assessment
            "Overlap / positivity assessment:",
            "  - Compute e_min=min(e_hat_i) and e_max=max(e_hat_i) using NON-missing e_hat values only.",
            "  - status='violation' if any e_hat_i <= 0.001 or >= 0.999 (or if ALL e_hat are missing).",
            "  - else status='warning' if any e_hat_i <= 0.01 or >= 0.99.",
            "  - else status='ok'.",
            "  - Write overlap_assessment.notes describing the reason (e.g., extreme propensities / weak overlap).",

            # Uncertainty/CI
            "Uncertainty:",
            "  - Unless an explicit variance/SE recipe is provided in the dataset (it usually is not), set se_hat=null.",
            "  - In that case set ci_95=[null, null] (IMPORTANT: ci_95 must always be a 2-element list).",
            "  - Do NOT invent bootstrap results or standard errors."
        ]

    else:
        raise ValueError("protocol must be one of: 'A', 'B1', 'B2', 'C', 'D'")

    # task instructions
    instructions = []
    instructions.append("TASK_INSTRUCTIONS")
    instructions.append("You are given observational data about covariates X, treatment A (0/1), and outcome Y.")
    instructions.append("Estimate the Average Treatment Effect (ATE).")
    instructions.append("Assume unconfoundedness given X, and discuss overlap/positivity based on the provided info.")
    instructions.append("You MUST NOT fabricate numbers. Use only the provided EVALUATION_DATASET block.")
    instructions.append("Return exactly one JSON object (no extra text).")
    instructions.extend(proto)
    instructions.append("END_TASK_INSTRUCTIONS")

    # few-shot examples
    examples_block = []
    if few_shot_examples:
        examples_block.append("FEW_SHOT_EXAMPLES_START")
        for i, ex in enumerate(few_shot_examples, start=1):
            examples_block.append(f"EXAMPLE_{i}_DATASET_START")
            examples_block.append(ex["dataset"].strip())
            examples_block.append(f"EXAMPLE_{i}_DATASET_END")
            examples_block.append(f"EXAMPLE_{i}_ANSWER_START")
            examples_block.append(json.dumps(ex["answer_json"], ensure_ascii=False))
            examples_block.append(f"EXAMPLE_{i}_ANSWER_END")
        examples_block.append("FEW_SHOT_EXAMPLES_END")

    # evaluation dataset
    eval_block = []
    eval_block.append("EVALUATION_DATASET_START")
    eval_block.append(serialized_data.strip())
    eval_block.append("EVALUATION_DATASET_END")

    # output schema reminder
    schema_block = []
    if include_schema_reminder:
        schema_block.append("OUTPUT_SCHEMA")
        schema_block.append("Return exactly one JSON object matching this schema. If cannot compute a field, set it to null (do NOT guess).")
        schema_block.append(json.dumps(MODULE2_OUTPUT_SCHEMA, ensure_ascii=False, indent=2))
        schema_block.append("Rules:")
        schema_block.append("1) Do not add extra keys.")
        schema_block.append("2) Do not output text outside JSON.")
        schema_block.append("3) If ci_95 is present (not null), it must satisfy ci_95[0] <= ate_hat <= ci_95[1].")
        schema_block.append("END_OUTPUT_SCHEMA")

    # assemble one final prompt document
    blocks = []
    blocks.extend(instructions)
    if examples_block:
        blocks.append("")
        blocks.extend(examples_block)
    blocks.append("")
    blocks.extend(eval_block)
    if schema_block:
        blocks.append("")
        blocks.extend(schema_block)

    return "\n".join(blocks)
