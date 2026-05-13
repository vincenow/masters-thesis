"""
Generate AI-enriched EuroVoc label descriptions using the Claude API.

For each EuroVoc concept used in MultiEURLEX, generates a 2-3 sentence
description in English, French, Dutch, and German, grounded in EU law and policy.

Output: generated_descriptors.json
Format mirrors eurovoc_descriptors.json but contains only the 4 target languages
and generated descriptions instead of the original short phrases.

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
        --descriptors path/to/eurovoc_descriptors.json \
        --output generated_descriptors.json \
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import anthropic
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
    """Load MultiEURLEX (English) and return all EuroVoc IDs used across all splits."""
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


# ── API call ──────────────────────────────────────────────────────────────────

async def generate_description(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    eurovoc_id: str,
    english_descriptor: str,
    lang_code: str,
    language_name: str,
    retries: int = 5,
) -> tuple[str, str, str | None]:
    """
    Returns (eurovoc_id, lang_code, description_or_None).
    Retries up to `retries` times on transient errors.
    """
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": build_user_prompt(english_descriptor, language_name),
                        }
                    ],
                )
            raw = response.content[0].text.strip()
            parsed = json.loads(raw)
            return eurovoc_id, lang_code, parsed["description"]
        except (json.JSONDecodeError, KeyError):
            if attempt < retries - 1:
                await asyncio.sleep(1)
            else:
                print(f"  [WARN] Malformed JSON for {eurovoc_id}/{lang_code}: {raw[:80]}")
                return eurovoc_id, lang_code, None
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s
            print(f"  [WARN] Rate limit hit for {eurovoc_id}/{lang_code}, waiting {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
            else:
                print(f"  [ERROR] {eurovoc_id}/{lang_code}: {e}")
                return eurovoc_id, lang_code, None
    return eurovoc_id, lang_code, None


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(descriptors_path: str, output_path: str, concurrency: int):
    # Load existing output if present (allows resuming interrupted runs)
    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file) as f:
            results: dict = json.load(f)
        print(f"Resuming: {len(results)} concepts already have at least one language done.")
    else:
        results = {}

    # Collect used label IDs from the dataset
    used_ids = get_used_label_ids()

    # Load EuroVoc descriptors
    with open(descriptors_path) as f:
        eurovoc: dict = json.load(f)

    # Build task list — skip already completed (concept, language) pairs
    todo = []
    for eid in used_ids:
        if eid not in eurovoc:
            continue
        en_desc = eurovoc[eid].get("en")
        if not en_desc:
            continue
        for lang_code, lang_name in LANGUAGES.items():
            if eid in results and lang_code in results[eid]:
                continue
            todo.append((eid, en_desc, lang_code, lang_name))

    total = len(todo)
    concepts_remaining = total // len(LANGUAGES)
    print(f"Tasks remaining: {total} (~{concepts_remaining} concepts × {len(LANGUAGES)} languages)")
    if total == 0:
        print("Nothing to do.")
        return

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    completed = 0
    save_every = 200
    start = time.time()

    tasks = [
        generate_description(client, semaphore, eid, en_desc, lc, ln)
        for eid, en_desc, lc, ln in todo
    ]

    for coro in asyncio.as_completed(tasks):
        eurovoc_id, lang_code, description = await coro
        completed += 1

        if description is not None:
            if eurovoc_id not in results:
                results[eurovoc_id] = {}
            results[eurovoc_id][lang_code] = description

        if completed % 100 == 0 or completed == total:
            elapsed = time.time() - start
            rate = completed / elapsed
            remaining = (total - completed) / rate if rate > 0 else 0
            print(
                f"  {completed}/{total} done "
                f"({rate:.1f} calls/s, ~{remaining/60:.1f} min remaining)"
            )

        if completed % save_every == 0 or completed == total:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\nDone. Output written to {output_path}")
    failed = sum(
        1 for eid, _, lc, _ in todo
        if eid not in results or lc not in results.get(eid, {})
    )
    if failed:
        print(f"  {failed} tasks failed — re-run to retry.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EuroVoc label descriptions.")
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
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel API calls (default: 10)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    asyncio.run(
        main(
            descriptors_path=args.descriptors,
            output_path=args.output,
            concurrency=args.concurrency,
        )
    )