"""Run Module 4 prompts as individual API requests."""

import json
import os
from datetime import datetime

from .api_client import MAX_RETRIES, run_one


DEFAULT_OUTPUT_PATH = "results/individual_results.jsonl"


def load_completed_keys(path):
    """Return (prompt_id, model) pairs already written to a JSONL output file."""
    done = set()

    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            done.add((record["prompt_id"], record["model"]))

    return done


def run_grid(simulations, gpt_model, out_path=DEFAULT_OUTPUT_PATH, client=None):
    """Run every prompt in a simulation grid, resuming completed jobs."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_completed_keys(out_path)

    with open(out_path, "a", encoding="utf-8") as f:
        for sim in simulations:
            prompt_id = sim["prompt_id"]
            base_prompt = sim["base_prompt"]
            tau = sim["tau"]

            key = (prompt_id, gpt_model)
            if key in done:
                continue

            record = {
                "prompt_id": prompt_id,
                "model": gpt_model,
                "tau": tau,
                "prompt": base_prompt,
                "timestamp": datetime.utcnow().isoformat(),
            }

            try:
                output_json, usage, retries = run_one(
                    gpt_model,
                    base_prompt,
                    client=client,
                )
                record.update(
                    {
                        "output": output_json,
                        "usage": usage,
                        "retry_count": retries,
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "output": None,
                        "usage": None,
                        "retry_count": MAX_RETRIES,
                        "status": "error",
                        "error": str(exc),
                    }
                )

            f.write(json.dumps(record) + "\n")
            f.flush()
            print("Completed:", key)
