"""Build, submit, monitor, and download Module 4 batch jobs."""

import json
import time
from pathlib import Path

from .api_client import STRUCTURED_SCHEMA, get_client


def build_batch_file(
    simulations,
    gpt_model,
    file_path="results/batch_input.jsonl",
):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for sim in simulations:
            request = {
                "custom_id": sim["prompt_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": gpt_model,
                    "input": sim["base_prompt"],
                    "temperature": 0,
                    "max_output_tokens": 500,
                    "text": {"format": STRUCTURED_SCHEMA},
                },
            }
            f.write(json.dumps(request) + "\n")

    print("Batch file successfully created:", file_path)
    return file_path


def submit_batch(batch_file_path, client=None):
    if client is None:
        client = get_client()

    with open(batch_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    print("Batch submitted.")
    print("Batch ID:", batch.id)
    return batch.id


def wait_for_batch(batch_id, poll_interval=30, client=None):
    if client is None:
        client = get_client()

    while True:
        batch = client.batches.retrieve(batch_id)
        print("Status:", batch.status)

        if batch.errors:
            print("Errors:", batch.errors)

        if batch.status in ["completed", "failed", "cancelled"]:
            return batch

        time.sleep(poll_interval)


def download_batch_results(
    batch,
    output_path="results/batch_output.jsonl",
    error_path="results/batch_errors.jsonl",
    client=None,
):
    if client is None:
        client = get_client()

    output_path = Path(output_path)
    error_path = Path(error_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    print("Output file:", batch.output_file_id)
    print("Error file:", batch.error_file_id)
    print("Request counts:", batch.request_counts)

    if batch.output_file_id:
        result = client.files.content(batch.output_file_id)
        output_path.write_bytes(result.read())
        print("Results saved to", output_path)

    if batch.error_file_id:
        result = client.files.content(batch.error_file_id)
        error_path.write_bytes(result.read())
        print("Errors saved to", error_path)
