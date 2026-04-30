import json
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
import bm25s
from FlagEmbedding import FlagReranker

TEST_MODE = False

CANDIDATE_K = 100

url = "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json"
eurovoc_concepts = requests.get(url).json()

print("Loading reranker model...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device='cuda')


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

    # Filter out None/empty descriptors and keep index mapping
    valid_indices = [i for i, d in enumerate(label_descriptors_raw) if d and d.strip()]
    valid_descriptors = [label_descriptors_raw[i] for i in valid_indices]

    if len(valid_indices) < len(label_descriptors_raw):
        print(f"Warning: {len(label_descriptors_raw) - len(valid_indices)} labels had no "
              f"'{label_lang}' descriptor and were excluded.")

    print("Building BM25S index over labels...")
    tokenized_labels = bm25s.tokenize(valid_descriptors, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokenized_labels)

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

        # Stage 1: BM25S retrieval of top CANDIDATE_K
        tokenized_query = bm25s.tokenize([doc['text']], stopwords="en")
        retrieved_indices, _ = retriever.retrieve(tokenized_query, k=min(CANDIDATE_K, len(valid_indices)))
        # retrieved_indices[0] are positions into valid_descriptors
        top_candidate_positions = retrieved_indices[0].tolist()

        # Stage 2: rerank with cross-encoder using the valid descriptor text
        doc_text = doc['text']
        pairs = [[doc_text, valid_descriptors[pos]] for pos in top_candidate_positions]
        rerank_scores = np.array(reranker.compute_score(pairs, batch_size=32))
        reranked_order = np.argsort(rerank_scores)[::-1]

        # Map back through valid_indices to get original label integer IDs
        reranked_predictions = [valid_indices[top_candidate_positions[i]] for i in reranked_order]

        for k in k_values:
            results['precision'][k].append(precision_at_k(true_labels, reranked_predictions, k))
            results['recall'][k].append(recall_at_k(true_labels, reranked_predictions, k))
            results['ndcg'][k].append(ndcg_at_k(true_labels, reranked_predictions, k))

    print(f"\nResults for BM25 + BGE-M3 reranker | {language_name} | {label_condition}")
    for k in k_values:
        print(f"  k={k}: P={np.mean(results['precision'][k]):.4f} "
              f"R={np.mean(results['recall'][k]):.4f} "
              f"NDCG={np.mean(results['ndcg'][k]):.4f}")

    results_to_save = {
        'model': 'BM25 + bge-reranker-v2-m3',
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
    filename = f'{prefix}results_bm25_reranked_{language}_{label_condition.replace(" ", "_").lower()}.json'
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved: {filename}")


if TEST_MODE:
    conditions = [('en', 'English', 'en', 'EN labels')]
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