"""
Generate AI-enriched EuroVoc label descriptions using the Anthropic Batch API.

Submits all 4,591 concepts × 4 languages = 18,364 requests in a single batch,
polls until complete, then saves results to generated_descriptors.json.

Output format mirrors eurovoc_descriptors.json:
{
    "100149": {
        "en": "Social questions refer to ...",
        "fr": "Les questions sociales ...",
        "de": "Soziale Fragen umfassen ...",
        "nl": "Sociale vraagstukken omvatten ..."
    },
    ...
}

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here

    python generate_descriptors.py \
        --descriptors /home/ubuntu/masters-thesis/eurovoc_descriptors.json \
        --output generated_descriptors.json
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


# ── Label ID collection ───────────────────────────────────────────────────────

def get_used_label_ids() -> set[str]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(descriptors_path: str, output_path: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic()

    used_ids = get_used_label_ids()

    with open(descriptors_path) as f:
        eurovoc: dict = json.load(f)

    # Build one request per (concept, language) pair
    # custom_id format: "{eurovoc_id}__{lang_code}" — used to parse results later
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
        for lang_code, lang_name in LANGUAGES.items():
            requests.append(
                Request(
                    custom_id=f"{eid}__{lang_code}",
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
            )

    print(f"Prepared {len(requests)} requests ({skipped} concepts skipped).")

    # Submit batch
    print("Submitting batch...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted. ID: {batch.id}")

    # Save batch ID in case retrieval is interrupted
    with open("batch_id.txt", "w") as f:
        f.write(batch.id)

    # Poll until done
    print("Polling for completion (every 60s)...")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
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

    # Parse results
    results = {}
    failed = []

    for result in client.messages.batches.results(batch.id):
        custom_id = result.custom_id  # format: "{eurovoc_id}__{lang_code}"
        eurovoc_id, lang_code = custom_id.rsplit("__", 1)

        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text.strip()
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

    # Save output
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
    args = parser.parse_args()
    main(args.descriptors, args.output)