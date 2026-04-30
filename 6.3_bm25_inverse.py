import json
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
import bm25s

TEST_MODE = False

url = "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json"
eurovoc_concepts = requests.get(url).json()


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
    label_descriptors = [eurovoc_concepts[label_id][label_lang] for label_id in label_ids]

    # Collect all document texts for indexing
    print("Collecting documents...")
    all_texts = [doc['text'] for doc in dataset]

    # Index documents as the corpus (natural BM25 direction)
    print("Building BM25S index over documents...")
    tokenized_docs = bm25s.tokenize(all_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokenized_docs)

    # For each label, query the document corpus and record the score
    # that label gives to each document. Result: a (num_labels, num_docs) score matrix.
    print("Querying with each label descriptor...")
    num_docs = len(all_texts)
    num_labels = len(label_descriptors)

    # label_doc_scores[label_idx, doc_idx] = BM25 score of label query against that document
    label_doc_scores = np.zeros((num_labels, num_docs), dtype=np.float32)

    for label_idx, descriptor in enumerate(tqdm(label_descriptors, desc="Label queries")):
        if not descriptor or not descriptor.strip():
            continue
        tokenized_query = bm25s.tokenize([descriptor], stopwords="en")
        # Retrieve all documents (k=num_docs) to get full ranking
        retrieved_indices, retrieved_scores = retriever.retrieve(
            tokenized_query, k=num_docs)
        for pos, (doc_idx, score) in enumerate(
                zip(retrieved_indices[0].tolist(), retrieved_scores[0].tolist())):
            label_doc_scores[label_idx, doc_idx] = score

    # Now evaluate per document: rank labels by the score they gave that document
    k_values = [5, 10, 20, 50, 100]
    results = {
        'precision': {k: [] for k in k_values},
        'recall':    {k: [] for k in k_values},
        'ndcg':      {k: [] for k in k_values}
    }

    print("Computing per-document metrics...")
    for doc_idx, doc in enumerate(tqdm(dataset, desc="Evaluating")):
        true_labels = set(doc['labels'])
        if len(true_labels) == 0:
            continue

        # Rank all labels by the score they gave this document
        doc_scores = label_doc_scores[:, doc_idx]
        ranked_predictions = np.argsort(doc_scores)[::-1].tolist()

        for k in k_values:
            results['precision'][k].append(precision_at_k(true_labels, ranked_predictions, k))
            results['recall'][k].append(recall_at_k(true_labels, ranked_predictions, k))
            results['ndcg'][k].append(ndcg_at_k(true_labels, ranked_predictions, k))

    print(f"\nResults for BM25 (label queries) | {language_name} | {label_condition}")
    for k in k_values:
        print(f"  k={k}: P={np.mean(results['precision'][k]):.4f} "
              f"R={np.mean(results['recall'][k]):.4f} "
              f"NDCG={np.mean(results['ndcg'][k]):.4f}")

    results_to_save = {
        'model': 'BM25 (label queries)',
        'dataset': 'MultiEURLEX',
        'language': f'{language_name} ({label_condition})',
        'test_mode': TEST_MODE,
        'num_documents': len(dataset),
        'num_labels': num_labels,
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
    filename = f'{prefix}results_bm25_labelquery_{language}_{label_condition.replace(" ", "_").lower()}.json'
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