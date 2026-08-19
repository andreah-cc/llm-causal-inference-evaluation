"""Parse OpenAI Batch API JSONL output into flat dictionaries."""

import json


def extract_batch_results(jsonl_path):
    results = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            custom_id = row.get("custom_id")
            body = row.get("response", {}).get("body", {})
            model = body.get("model")

            parsed_output = None
            try:
                output_text = body["output"][0]["content"][0]["text"]
                parsed_output = json.loads(output_text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                parsed_output = None

            result_row = {
                "custom_id": custom_id,
                "model": model,
            }
            if isinstance(parsed_output, dict):
                result_row.update(parsed_output)
            else:
                result_row["format_failure"] = True

            results.append(result_row)

    return results
