import json
import numpy as np
import requests
import os
import time
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ─────────────────────────────────────────────────────────────

TEST_MODE     = False
CENTROIDS_DIR = '.'  # directory where centroids_*.npy files are saved
OUTPUT_DIR    = '.'  # directory where result JSON files will be saved

MODELS = [
    {
        'short':          'bge-m3',
        'name':           'BAAI/bge-m3',
        'batch_size':     8,
        'prefix':         None,
        'prompt_name':    None,
        'kwargs':         {'device': 'cuda'},
        'max_seq_length': 512,
    },
    {
        'short':          'e5-small',
        'name':           'intfloat/multilingual-e5-small',
        'batch_size':     32,
        'prefix':         'query: ',  # E5 uses query prefix for documents at eval time
        'prompt_name':    None,
        'kwargs':         {'device': 'cuda'},
        'max_seq_length': None,
    },
    {
        'short':          'labse',
        'name':           'sentence-transformers/LaBSE',
        'batch_size':     32,
        'prefix':         None,
        'prompt_name':    None,
        'kwargs':         {'device': 'cuda'},
        'max_seq_length': None,
    },
    {
        'short':          'qwen3-embedding-0.6b',
        'name':           'Qwen/Qwen3-Embedding-0.6B',
        'batch_size':     8,
        'prefix':         None,
        'prompt_name':    'query',    # query side at eval time
        'kwargs':         {'device': 'cuda'},
        'max_seq_length': 512,
    },
    {
        'short':          'harrier-oss-v1-0.6b',
        'name':           'microsoft/harrier-oss-v1-0.6b',
        'batch_size':     8,
        'prefix':         None,
        'prompt_name':    'web_search_query',  # query side at eval time
        'kwargs':         {'device': 'cuda', 'model_kwargs': {'dtype': 'auto'}},
        'max_seq_length': 512,
    },
]

INCLUDE_OPENAI = True
OPENAI_MODEL   = 'text-embedding-3-small'
OPENAI_SHORT   = 'openai'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

languages = [
    ('en', 'English'),
    ('fr', 'French'),
    ('nl', 'Dutch'),
    ('de', 'German'),
]

# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Evaluation + saving ───────────────────────────────────────────────────────

def evaluate_and_save(model_short, doc_embeddings, centroid_embeddings,
                      dataset, language, language_name):
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

        similarities = cosine_similarity([doc_embeddings[idx]], centroid_embeddings)[0]
        ranked_predictions = np.argsort(similarities)[::-1]

        for k in k_values:
            results['precision'][k].append(precision_at_k(true_labels, ranked_predictions, k))
            results['recall'][k].append(recall_at_k(true_labels, ranked_predictions, k))
            results['ndcg'][k].append(ndcg_at_k(true_labels, ranked_predictions, k))

    print(f"\nResults for {model_short} | {language_name} | centroids")
    for k in k_values:
        print(f"  k={k}: P={np.mean(results['precision'][k]):.4f} "
              f"R={np.mean(results['recall'][k]):.4f} "
              f"NDCG={np.mean(results['ndcg'][k]):.4f}")

    results_to_save = {
        'model': model_short,
        'dataset': 'MultiEURLEX',
        'language': f'{language_name} (centroids)',
        'test_mode': TEST_MODE,
        'num_documents': len(dataset),
        'num_labels': centroid_embeddings.shape[0],
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
    filename = f"{OUTPUT_DIR}/{prefix}results_{model_short}_{language}_centroids.json"
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved: {filename}")


# ── SentenceTransformer models ────────────────────────────────────────────────

for model_cfg in MODELS:
    short = model_cfg['short']
    centroid_path = f"{CENTROIDS_DIR}/centroids_{short}.npy"

    if not os.path.exists(centroid_path):
        print(f"\nSkipping {short}: centroid file not found at {centroid_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Model: {short}")
    print(f"{'='*60}")

    print("Loading centroid embeddings...")
    centroid_embeddings = np.load(centroid_path)
    print(f"  Centroid shape: {centroid_embeddings.shape}")

    print("Loading model...")
    model = SentenceTransformer(model_cfg['name'], **model_cfg['kwargs'])
    if model_cfg['max_seq_length']:
        model.max_seq_length = model_cfg['max_seq_length']

    for language, language_name in languages:
        print(f"\n--- Language: {language_name} ---")

        dataset = load_dataset(
            'coastalcph/multi_eurlex', language,
            split='test',
            label_level='all_levels',
            trust_remote_code=True
        )

        if TEST_MODE:
            dataset = dataset.select(range(50))
            print("TEST MODE: using 50 documents only")

        texts = [doc['text'] for doc in dataset]
        if model_cfg['prefix']:
            texts = [model_cfg['prefix'] + t for t in texts]

        print("Encoding documents...")
        doc_embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=model_cfg['batch_size'],
            normalize_embeddings=True,
            prompt_name=model_cfg['prompt_name'],
        )

        evaluate_and_save(short, doc_embeddings, centroid_embeddings,
                          dataset, language, language_name)

    del model, doc_embeddings
    print(f"\nDone with {short}, model unloaded from GPU.")

# ── OpenAI ────────────────────────────────────────────────────────────────────

if INCLUDE_OPENAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Run: export OPENAI_API_KEY='sk-...'"
        )

    import openai

    print(f"\n{'='*60}")
    print(f"Model: {OPENAI_SHORT}")
    print(f"{'='*60}")

    centroid_path = f"{CENTROIDS_DIR}/centroids_{OPENAI_SHORT}.npy"
    print("Loading centroid embeddings...")
    centroid_embeddings = np.load(centroid_path)
    print(f"  Centroid shape: {centroid_embeddings.shape}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    encoder = tiktoken.encoding_for_model(OPENAI_MODEL)

    def truncate(text, max_tokens=8000):
        tokens = encoder.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoder.decode(tokens)

    def get_openai_embeddings(texts, batch_size=5):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = [truncate(t) for t in texts[i:i + batch_size]]
            while True:
                try:
                    response = client.embeddings.create(input=batch, model=OPENAI_MODEL)
                    all_embeddings.extend([item.embedding for item in response.data])
                    time.sleep(0.5)
                    break
                except openai.RateLimitError:
                    print("Rate limit hit, waiting 10s...")
                    time.sleep(10)
        return np.array(all_embeddings, dtype=np.float32)

    for language, language_name in languages:
        print(f"\n--- Language: {language_name} ---")

        dataset = load_dataset(
            'coastalcph/multi_eurlex', language,
            split='test',
            label_level='all_levels',
            trust_remote_code=True
        )

        if TEST_MODE:
            dataset = dataset.select(range(50))
            print("TEST MODE: using 50 documents only")

        # Use cached doc embeddings if available (same cache as existing OpenAI scripts)
        doc_cache = f'openai_doc_embeddings_{language}.npy'
        if os.path.exists(doc_cache):
            print(f"Loading cached document embeddings ({doc_cache})...")
            doc_embeddings = np.load(doc_cache)
        else:
            print("Encoding documents via OpenAI API...")
            texts = [doc['text'] for doc in dataset]
            doc_embeddings = get_openai_embeddings(texts)
            np.save(doc_cache, doc_embeddings)
        print(f"  Document embeddings shape: {doc_embeddings.shape}")

        evaluate_and_save(OPENAI_SHORT, doc_embeddings, centroid_embeddings,
                          dataset, language, language_name)

print("\nAll conditions complete!")