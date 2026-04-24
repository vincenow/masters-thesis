import os
import json
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagReranker

TEST_MODE = False

CANDIDATE_K = 100
MODEL_NAME = 'BAAI/bge-m3'
MODEL_SHORT = 'bge-m3'

url = "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json"
eurovoc_concepts = requests.get(url).json()

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME, device='cuda')
embedding_model.max_seq_length = 512

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

    label_embeddings = embedding_model.encode(
        label_descriptors_raw,
        show_progress_bar=True,
        batch_size=32,
    )

    doc_embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=8,
    )

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
        rerank_scores = np.array(reranker.compute_score(pairs, batch_size=4))
        reranked_order = np.argsort(rerank_scores)[::-1]
        reranked_predictions = top_candidate_indices[reranked_order]

        for k in k_values:
            results['precision'][k].append(precision_at_k(true_labels, reranked_predictions, k))
            results['recall'][k].append(recall_at_k(true_labels, reranked_predictions, k))
            results['ndcg'][k].append(ndcg_at_k(true_labels, reranked_predictions, k))

    print(f"\nResults for {MODEL_SHORT} + BGE-M3 reranker | {language_name} | {label_condition}")
    for k in k_values:
        print(f"  k={k}: P={np.mean(results['precision'][k]):.4f} "
              f"R={np.mean(results['recall'][k]):.4f} "
              f"NDCG={np.mean(results['ndcg'][k]):.4f}")

    results_to_save = {
        'model': f'{MODEL_SHORT} + bge-reranker-v2-m3',
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
    filename = f'{prefix}results_bge_m3_reranked_{language}_{label_condition.replace(" ", "_").lower()}.json'
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
