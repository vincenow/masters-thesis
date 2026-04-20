import os
import json
import time
import numpy as np
import requests
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEST_MODE = True  # set to True for quick test run

CANDIDATE_K = 100
OPENAI_MODEL = "text-embedding-3-small"

url = "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json"
eurovoc_concepts = requests.get(url).json()

print("Loading reranker model...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


# ── OpenAI embedding helpers ──────────────────────────────────────────────────

def truncate_text(text, max_tokens=8000):
    encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoder.decode(tokens)


def get_openai_embeddings(texts, batch_size=5):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = [truncate_text(t) for t in texts[i:i + batch_size]]
        try:
            response = client.embeddings.create(input=batch, model=OPENAI_MODEL)
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            all_embeddings.extend([[0.0] * 1536] * len(batch))
        time.sleep(0.5)
    return np.array(all_embeddings)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def precision_at_k(y_true, y_pred, k):
    top_k = y_pred[:k]
    relevant = sum(1 for label in top_k if label in y_true)
    return relevant / k


def recall_at_k(y_true, y_pred, k):
    if len(y_true) == 0:
        return 0.0
    top_k = y_pred[:k]
    relevant = sum(1 for label in top_k if label in y_true)
    return relevant / len(y_true)


def ndcg_at_k(y_true, y_pred, k):
    top_k = y_pred[:k]
    dcg = sum((1 if label in y_true else 0) / np.log2(i + 2)
              for i, label in enumerate(top_k))
    ideal_k = min(len(y_true), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


# ── Main condition runner ─────────────────────────────────────────────────────

def run_condition(language, language_name, label_lang, label_condition):
    print(f"\n{'='*60}")
    print(f"Running: {language_name} docs | {label_condition}")
    print(f"{'='*60}")

    dataset = load_dataset('coastalcph/multi_eurlex', language, split='test',
                           label_level='all_levels', trust_remote_code=True)

    if TEST_MODE:
        dataset = dataset.select(range(50))
        print("TEST MODE: using 50 documents only")

    classlabel = dataset.features["labels"].feature
    label_ids = classlabel.names
    label_descriptors_raw = [eurovoc_concepts[label_id][label_lang] for label_id in label_ids]

    # ── Label embeddings (cached per language) ────────────────────────────────
    label_cache = f'openai_label_embeddings_{label_lang}.npy'
    if os.path.exists(label_cache):
        print(f"Loading cached label embeddings ({label_cache})...")
        label_embeddings = np.load(label_cache)
    else:
        print(f"Encoding {label_lang} label embeddings...")
        label_embeddings = get_openai_embeddings(label_descriptors_raw)
        np.save(label_cache, label_embeddings)
    print(f"Label embeddings shape: {label_embeddings.shape}")

    # ── Document embeddings (cached per language) ─────────────────────────────
    doc_cache = f'openai_doc_embeddings_{language}.npy'
    if os.path.exists(doc_cache):
        print(f"Loading cached document embeddings ({doc_cache})...")
        doc_embeddings = np.load(doc_cache)
    else:
        print(f"Encoding {language_name} document embeddings...")
        texts = [doc['text'] for doc in dataset]
        doc_embeddings = get_openai_embeddings(texts)
        np.save(doc_cache, doc_embeddings)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    k_values = [5, 10, 20, 50, 100]
    results = {
        'precision': {k: [] for k in k_values},
        'recall':    {k: [] for k in k_values},
        'ndcg':      {k: [] for k in k_values}
    }

    for idx, doc in enumerate(tqdm(dataset, desc="Evaluating")):
        true_labels = set(doc['labels'])
        if len(true_labels) == 0:
            continue

        # Stage 1: retrieve top CANDIDATE_K by cosine similarity
        similarities = cosine_similarity([doc_embeddings[idx]], label_embeddings)[0]
        top_candidate_indices = np.argsort(similarities)[::-1][:CANDIDATE_K]

        # Stage 2: rerank with cross-encoder
        doc_text = doc['text']
        pairs = [[doc_text, label_descriptors_raw[i]] for i in top_candidate_indices]
        rerank_scores = np.array(reranker.compute_score(pairs, batch_size=32))
        reranked_order = np.argsort(rerank_scores)[::-1]
        reranked_predictions = top_candidate_indices[reranked_order]

        for k in k_values:
            results['precision'][k].append(precision_at_k(true_labels, reranked_predictions, k))
            results['recall'][k].append(recall_at_k(true_labels, reranked_predictions, k))
            results['ndcg'][k].append(ndcg_at_k(true_labels, reranked_predictions, k))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nResults for OpenAI {OPENAI_MODEL} + BGE-M3 reranker | {language_name} | {label_condition}")
    for k in k_values:
        print(f"  k={k}: P={np.mean(results['precision'][k]):.4f} "
              f"R={np.mean(results['recall'][k]):.4f} "
              f"NDCG={np.mean(results['ndcg'][k]):.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_to_save = {
        'model': f'{OPENAI_MODEL} + bge-reranker-v2-m3',
        'dataset': 'MultiEURLEX',
        'language': f'{language_name} ({label_condition})',
        'candidate_k': CANDIDATE_K,
        'test_mode': TEST_MODE,
        'num_documents': len(dataset),
        'num_labels': len(label_descriptors_raw),
        'metrics': {
            metric_name: {
                k: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'values': [float(x) for x in scores]
                }
                for k, scores in metric_data.items()
            }
            for metric_name, metric_data in results.items()
        }
    }

    prefix = 'TEST_' if TEST_MODE else ''
    filename = f'{prefix}results_openai_reranked_{language}_{label_condition.replace(" ", "_").lower()}.json'
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved: {filename}")


# ── Conditions ────────────────────────────────────────────────────────────────

if TEST_MODE:
    conditions = [
        ('en', 'English', 'en', 'EN labels'),
    ]
else:
    conditions = [
        ('en', 'English', 'en', 'EN labels'),
        ('fr', 'French',  'en', 'EN labels'),
        ('nl', 'Dutch',   'en', 'EN labels'),
        ('de', 'German',  'en', 'EN labels'),
        ('fr', 'French',  'fr', 'native labels'),
        ('nl', 'Dutch',   'nl', 'native labels'),
        ('de', 'German',  'de', 'native labels'),
    ]

for language, language_name, label_lang, label_condition in conditions:
    run_condition(language, language_name, label_lang, label_condition)

print("\nAll conditions complete!")