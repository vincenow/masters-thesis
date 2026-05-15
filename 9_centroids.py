import json
import numpy as np
import requests
import random
import os
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = []

INCLUDE_OPENAI = True
OPENAI_MODEL   = 'text-embedding-3-small'
OPENAI_SHORT   = 'openai'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_DOCS_PER_LABEL = 10
RANDOM_SEED        = 42
LANGUAGE           = 'en'   # English training documents only (Option A)
OUTPUT_DIR         = '.'    # Save centroids here

# ── Load training data ─────────────────────────────────────────────────────────

print("Loading English training split...")
train_dataset = load_dataset(
    'coastalcph/multi_eurlex', LANGUAGE,
    split='train',
    label_level='all_levels',
    trust_remote_code=True
)

classlabel = train_dataset.features["labels"].feature
label_ids  = classlabel.names   # list of EuroVoc ID strings, length 7390
n_labels   = len(label_ids)
print(f"  {len(train_dataset)} training documents, {n_labels} labels")

# ── Build label → document index ───────────────────────────────────────────────
# Maps each label integer index → list of document integer indices

print("Building label-to-document index...")
label_to_doc_indices = defaultdict(list)
for doc_idx, doc in enumerate(tqdm(train_dataset, desc="Indexing")):
    for label_int in doc['labels']:
        label_to_doc_indices[label_int].append(doc_idx)

# Report coverage
n_covered = sum(1 for i in range(n_labels) if label_to_doc_indices[i])
print(f"  Labels with at least 1 training document: {n_covered} / {n_labels}")
print(f"  Labels with zero training documents: {n_labels - n_covered}")

# ── Sample up to MAX_DOCS_PER_LABEL doc indices per label ──────────────────────

rng = random.Random(RANDOM_SEED)
sampled_indices = {}   # label_int → list of doc indices (up to 10)
for label_int in range(n_labels):
    docs = label_to_doc_indices[label_int]
    if len(docs) == 0:
        sampled_indices[label_int] = []
    elif len(docs) <= MAX_DOCS_PER_LABEL:
        sampled_indices[label_int] = docs
    else:
        sampled_indices[label_int] = rng.sample(docs, MAX_DOCS_PER_LABEL)

# ── Collect unique documents to embed (avoid re-embedding same doc) ────────────

all_needed_doc_indices = sorted(set(
    idx for indices in sampled_indices.values() for idx in indices
))
print(f"\nUnique training documents to embed: {len(all_needed_doc_indices)}")

# Pre-fetch texts for needed documents only
print("Fetching document texts...")
doc_index_to_text = {
    idx: train_dataset[idx]['text']
    for idx in tqdm(all_needed_doc_indices, desc="Fetching texts")
}

# ── Helper: build centroids from embeddings dict ───────────────────────────────

def build_centroids(doc_embeddings_map, n_labels, sampled_indices, embed_dim):
    """
    doc_embeddings_map: dict {doc_idx: embedding vector}
    Returns centroid matrix of shape (n_labels, embed_dim).
    Labels with no training documents get a zero vector.
    """
    centroids = np.zeros((n_labels, embed_dim), dtype=np.float32)
    for label_int in range(n_labels):
        indices = sampled_indices[label_int]
        if not indices:
            continue  # zero vector fallback
        vecs = np.stack([doc_embeddings_map[i] for i in indices])
        centroids[label_int] = vecs.mean(axis=0)
    return centroids

# ── Embed with SentenceTransformer models ─────────────────────────────────────

texts_to_embed = [doc_index_to_text[i] for i in all_needed_doc_indices]

for model_cfg in MODELS:
    short = model_cfg['short']
    print(f"\n{'='*60}")
    print(f"Model: {short}")
    print(f"{'='*60}")

    print("Loading model...")
    model = SentenceTransformer(model_cfg['name'], **model_cfg['kwargs'])
    if model_cfg['encode_kwargs'].get('max_seq_length'):
        model.max_seq_length = model_cfg['encode_kwargs']['max_seq_length']

    # Apply prefix if needed (E5)
    if model_cfg['prefix']:
        prefixed_texts = [model_cfg['prefix'] + t for t in texts_to_embed]
    else:
        prefixed_texts = texts_to_embed

    print(f"Embedding {len(prefixed_texts)} documents...")
    embeddings = model.encode(
        prefixed_texts,
        show_progress_bar=True,
        batch_size=model_cfg['batch_size'],
        normalize_embeddings=True,
        prompt_name=model_cfg['prompt_name'],  # None for all models (passage side)
    )

    # Map back to doc index
    doc_embeddings_map = {
        doc_idx: embeddings[pos]
        for pos, doc_idx in enumerate(all_needed_doc_indices)
    }

    embed_dim = embeddings.shape[1]
    centroids = build_centroids(doc_embeddings_map, n_labels, sampled_indices, embed_dim)

    out_path = f"{OUTPUT_DIR}/centroids_{short}.npy"
    np.save(out_path, centroids)
    print(f"Saved: {out_path}  shape={centroids.shape}")

    del model, embeddings, doc_embeddings_map  # free GPU memory before next model

# ── OpenAI ─────────────────────────────────────────────────────────────────────

if INCLUDE_OPENAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Run: export OPENAI_API_KEY='sk-...'"
        )

    import openai
    import time

    print(f"\n{'='*60}")
    print(f"Model: {OPENAI_SHORT}")
    print(f"{'='*60}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def embed_openai_batch(texts, model=OPENAI_MODEL, batch_size=50):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI batches"):
            batch = texts[i:i+batch_size]
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [item.embedding for item in sorted(
                response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # rate limit headroom
        return np.array(all_embeddings, dtype=np.float32)

    print(f"Embedding {len(texts_to_embed)} documents via OpenAI API...")
    embeddings = embed_openai_batch(texts_to_embed)

    doc_embeddings_map = {
        doc_idx: embeddings[pos]
        for pos, doc_idx in enumerate(all_needed_doc_indices)
    }

    embed_dim = embeddings.shape[1]
    centroids = build_centroids(doc_embeddings_map, n_labels, sampled_indices, embed_dim)

    out_path = f"{OUTPUT_DIR}/centroids_{OPENAI_SHORT}.npy"
    np.save(out_path, centroids)
    print(f"Saved: {out_path}  shape={centroids.shape}")

# ── Save metadata ──────────────────────────────────────────────────────────────

metadata = {
    'language':               LANGUAGE,
    'split':                  'train',
    'max_docs_per_label':     MAX_DOCS_PER_LABEL,
    'random_seed':            RANDOM_SEED,
    'n_labels':               n_labels,
    'n_unique_docs_embedded': len(all_needed_doc_indices),
    'label_coverage': {
        'with_docs':    int(n_covered),
        'without_docs': int(n_labels - n_covered),
    },
    'label_doc_counts': {
        str(i): len(sampled_indices[i]) for i in range(n_labels)
    }
}

with open(f"{OUTPUT_DIR}/centroids_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\nSaved: centroids_metadata.json")
print("\nCentroid construction complete!")