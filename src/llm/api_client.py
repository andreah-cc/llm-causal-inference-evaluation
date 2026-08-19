"""OpenAI API client and structured-output schema for Module 4.

Authentication:
    Set OPENAI_API_KEY in your environment before running this module.

This replaces the original Google Colab `userdata` key-loading cell so the
public repository does not depend on Colab-specific secrets.
"""

import logging
import random
import time

from openai import OpenAI


MAX_RETRIES = 6
BASE_BACKOFF = 1.5

STRUCTURED_SCHEMA = {
    "type": "json_schema",
    "name": "module2_output",
    "schema": {
        "type": "object",
        "properties": {
            "ate_hat": {"type": ["number", "null"]},
            "se_hat": {"type": ["number", "null"]},
            "ci_95": {
                "type": "array",
                "items": {"type": ["number", "null"]},
                "minItems": 2,
                "maxItems": 2,
            },
            "estimator_name": {"type": "string"},
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "overlap_assessment": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["ok", "warning", "violation", "unknown"],
                    },
                    "e_min": {"type": ["number", "null"]},
                    "e_max": {"type": ["number", "null"]},
                    "notes": {"type": "string"},
                },
                "required": ["status", "e_min", "e_max", "notes"],
                "additionalProperties": False,
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "balance_summary": {"type": ["string", "null"]},
                    "missingness": {"type": ["string", "null"]},
                    "model_specification_risk": {"type": ["string", "null"]},
                },
                "required": [
                    "balance_summary",
                    "missingness",
                    "model_specification_risk",
                ],
                "additionalProperties": False,
            },
            "refusal_or_caution": {"type": ["string", "null"]},
        },
        "required": [
            "ate_hat",
            "se_hat",
            "ci_95",
            "estimator_name",
            "assumptions",
            "overlap_assessment",
            "diagnostics",
            "refusal_or_caution",
        ],
        "additionalProperties": False,
    },
}


def get_client():
    """Create an OpenAI client using OPENAI_API_KEY from the environment."""
    return OpenAI()


def run_one(model: str, prompt: str, client=None):
    """Run one structured-output request with exponential-backoff retries."""
    if client is None:
        client = get_client()

    logging.basicConfig(level=logging.INFO)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0,
                max_output_tokens=500,
                store=False,
                text={"format": STRUCTURED_SCHEMA},
            )

            output_json = resp.output_parsed
            usage = resp.usage.model_dump() if resp.usage else None
            return output_json, usage, attempt

        except Exception as exc:
            wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            logging.warning("Retry %s after error: %s", attempt + 1, exc)
            time.sleep(wait)

    raise RuntimeError("Max retries exceeded")
