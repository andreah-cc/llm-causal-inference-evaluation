"""Parse and validate structured JSON outputs from the LLM."""

import json
import numpy as np


def _float_or_nan(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_llm_output_module2(raw_output: str):
    out = {
        "format_failure": False,
        "parse_error": None,
        "parsed": None,
        "ate_hat": np.nan,
        "se_hat": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "estimator_name": None,
        "assumptions": None,
        "overlap_status": "unknown",
        "overlap_warning": False,
        "e_min": np.nan,
        "e_max": np.nan,
        "overlap_notes": None,
        "diagnostics": None,
        "refusal_or_caution": None
    }

    try:
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON is not an object")
        out["parsed"] = parsed
    except Exception as e:
        out["format_failure"] = True
        out["parse_error"] = str(e)
        return out

    # required top-level keys check
    required = ["ate_hat", "se_hat", "ci_95", "estimator_name", "assumptions",
                "overlap_assessment", "diagnostics", "refusal_or_caution"]
    missing = [k for k in required if k not in parsed]
    if missing:
        out["format_failure"] = True
        out["parse_error"] = f"Missing required keys: {missing}"
        return out

    # ate_hat, se_hat
    out["ate_hat"] = _float_or_nan(parsed.get("ate_hat"))
    out["se_hat"]  = _float_or_nan(parsed.get("se_hat"))

    # ci_95
    ci = parsed.get("ci_95")
    if isinstance(ci, list) and len(ci) == 2:
        out["ci95_lower"] = _float_or_nan(ci[0])
        out["ci95_upper"] = _float_or_nan(ci[1])
    else:
        out["format_failure"] = True
        out["parse_error"] = "ci_95 must be a list of length 2"
        return out

    # overlap_assessment
    oa = parsed.get("overlap_assessment")
    if not isinstance(oa, dict):
        out["format_failure"] = True
        out["parse_error"] = "overlap_assessment must be an object"
        return out

    out["overlap_status"] = str(oa.get("status", "unknown"))
    out["e_min"] = _float_or_nan(oa.get("e_min"))
    out["e_max"] = _float_or_nan(oa.get("e_max"))
    out["overlap_notes"] = oa.get("notes", None)
    out["overlap_warning"] = out["overlap_status"] in ["warning", "violation"]

    # estimator_name, assumptions, diagnostics, refusal_or_caution
    out["estimator_name"] = parsed.get("estimator_name")
    out["assumptions"] = parsed.get("assumptions")
    out["diagnostics"] = parsed.get("diagnostics")
    out["refusal_or_caution"] = parsed.get("refusal_or_caution")

    # CI sanity check if CI is numeric
    L, U, ate = out["ci95_lower"], out["ci95_upper"], out["ate_hat"]
    if np.isfinite(L) and np.isfinite(U) and np.isfinite(ate):
        if not (L <= ate <= U):
            out["format_failure"] = True
            out["parse_error"] = "Constraint violated: ci_95[0] <= ate_hat <= ci_95[1]"
            return out

    return out
