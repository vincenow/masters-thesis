"""
Generate AI-enriched EuroVoc label descriptions using the Anthropic Batch API.

Usage:
    export ANTHROPIC_API_KEY=your_key_here

    # Submit new batch and retrieve results:
    python 10_generate_label_descriptors.py \
        --descriptors /home/ubuntu/masters-thesis/eurovoc_descriptors.json \
        --output generated_descriptors.json

    # Retrieve results from an existing batch (skip resubmission):
    python 10_generate_label_descriptors.py \
        --descriptors /home/ubuntu/masters-thesis/eurovoc_descriptors.json \
        --output generated_descriptors.json \
        --batch-id msgbatch_01VdfmzakZcNZQLqpSM83hZu

    # Retry failed requests from a previous run:
    python 10_generate_label_descriptors.py \
        --descriptors /home/ubuntu/masters-thesis/eurovoc_descriptors.json \
        --output generated_descriptors.json \
        --retry-failed failed_requests.txt
"""

import argparse
import json
import os
import sys
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datasets import load_dataset

# ── Configuration ─────────────────────────────────────────────────────────────

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "nl": "Dutch",
}

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are an expert in EU law and policy. "
    "You generate concise label descriptions for the EuroVoc thesaurus, "
    "used to classify official EU legal documents."
)


def build_user_prompt(english_descriptor: str, language_name: str) -> str:
    return (
        f'The EuroVoc concept in English is: "{english_descriptor}"\n\n'
        f"Write a description of this concept in {language_name} in 2-3 sentences. "
        f"Define the concept and explain its scope within EU law and policy.\n\n"
        f'Respond with only a JSON object in this format: {{"description": "..."}}'
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_used_label_ids() -> set:
    print("Loading MultiEURLEX to collect used label IDs...")
    dataset = load_dataset(
        "coastalcph/multi_eurlex",
        "en",
        label_level="all_levels",
        trust_remote_code=True,
    )
    classlabel = dataset["train"].features["labels"].feature

    all_indices = set()
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        for doc in dataset[split_name]:
            all_indices.update(doc["labels"])

    used_ids = {classlabel.int2str(i) for i in all_indices}
    print(f"Found {len(used_ids)} EuroVoc IDs used across all splits.")
    return used_ids


def build_request(custom_id: str, en_desc: str, lang_code: str) -> Request:
    lang_name = LANGUAGES[lang_code]
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(en_desc, lang_name),
                }
            ],
        ),
    )


def submit_and_poll(client, requests):
    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Batch submitted. ID: {batch_id}")

    with open("batch_id.txt", "w") as f:
        f.write(batch_id)

    print("Polling for completion (every 60s)...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  {time.strftime('%H:%M:%S')} — "
            f"processing: {counts.processing}, "
            f"succeeded: {counts.succeeded}, "
            f"errored: {counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    return batch_id


def parse_results(client, batch_id):
    results = {}
    failed = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        eurovoc_id, lang_code = custom_id.rsplit("__", 1)

        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                parsed = json.loads(raw)
                description = parsed["description"]
                if eurovoc_id not in results:
                    results[eurovoc_id] = {}
                results[eurovoc_id][lang_code] = description
            except (json.JSONDecodeError, KeyError):
                print(f"  [WARN] Malformed JSON for {custom_id}: {raw[:80]}")
                failed.append(custom_id)
        else:
            print(f"  [WARN] Failed: {custom_id} — {result.result.type}")
            failed.append(custom_id)

    return results, failed


def save_output(results, output_path, failed):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nDone.")
    print(f"  Concepts saved: {len(results)}")
    print(f"  Total descriptions: {sum(len(v) for v in results.values())}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output: {output_path}")
    if failed:
        with open("failed_requests.txt", "w") as f:
            f.write("\n".join(failed))
        print(f"  Failed IDs written to failed_requests.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(descriptors_path: str, output_path: str, batch_id: str = None, retry_failed: str = None):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic()

    with open(descriptors_path) as f:
        eurovoc = json.load(f)

    if retry_failed:
        # Load failed IDs and existing output, submit small retry batch
        with open(retry_failed) as f:
            failed_ids = [line.strip() for line in f if line.strip()]
        print(f"Retrying {len(failed_ids)} failed requests...")

        if os.path.exists(output_path):
            with open(output_path) as f:
                existing_results = json.load(f)
        else:
            existing_results = {}

        requests = []
        for custom_id in failed_ids:
            eurovoc_id, lang_code = custom_id.rsplit("__", 1)
            if eurovoc_id not in eurovoc:
                continue
            en_desc = eurovoc[eurovoc_id].get("en")
            if not en_desc or lang_code not in LANGUAGES:
                continue
            requests.append(build_request(custom_id, en_desc, lang_code))

        batch_id = submit_and_poll(client, requests)
        new_results, still_failed = parse_results(client, batch_id)

        for eid, langs in new_results.items():
            if eid not in existing_results:
                existing_results[eid] = {}
            existing_results[eid].update(langs)

        save_output(existing_results, output_path, still_failed)

    elif batch_id:
        # Retrieve already-completed batch
        print(f"Retrieving existing batch: {batch_id}")
        print("Polling for completion (every 60s)...")
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(
                f"  {time.strftime('%H:%M:%S')} — "
                f"processing: {counts.processing}, "
                f"succeeded: {counts.succeeded}, "
                f"errored: {counts.errored}"
            )
            if batch.processing_status == "ended":
                break
            time.sleep(60)

        print("\nBatch finished. Parsing results...")
        results, failed = parse_results(client, batch_id)
        save_output(results, output_path, failed)

    else:
        # Fresh run — load dataset, build all requests, submit
        used_ids = get_used_label_ids()

        requests = []
        skipped = 0
        for eid in sorted(used_ids):
            if eid not in eurovoc:
                skipped += 1
                continue
            en_desc = eurovoc[eid].get("en")
            if not en_desc:
                skipped += 1
                continue
            for lang_code in LANGUAGES:
                requests.append(build_request(f"{eid}__{lang_code}", en_desc, lang_code))

        print(f"Prepared {len(requests)} requests ({skipped} concepts skipped).")
        batch_id = submit_and_poll(client, requests)

        print("\nBatch finished. Parsing results...")
        results, failed = parse_results(client, batch_id)
        save_output(results, output_path, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptors",
        required=True,
        help="Path to eurovoc_descriptors.json",
    )
    parser.add_argument(
        "--output",
        default="generated_descriptors.json",
        help="Output path (default: generated_descriptors.json)",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Existing batch ID to retrieve instead of submitting a new batch.",
    )
    parser.add_argument(
        "--retry-failed",
        default=None,
        help="Path to failed_requests.txt to retry only failed requests.",
    )
    args = parser.parse_args()
    main(args.descriptors, args.output, args.batch_id, args.retry_failed)