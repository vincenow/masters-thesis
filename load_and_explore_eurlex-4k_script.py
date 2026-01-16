# %% [markdown]
# ##### Load and inspect

# %%
import scipy.sparse as sp
import numpy as np
from collections import Counter

# Set the base directory
DATA_DIR = '/Users/vincent/masters-thesis/X-Transformer/datasets/Eurlex-4K'

print("="*80)
print("EURLEX-4K DATASET OVERVIEW")
print("="*80)

# Load the data
Y_train = sp.load_npz(f'{DATA_DIR}/Y.trn.npz')
Y_test = sp.load_npz(f'{DATA_DIR}/Y.tst.npz')
X_train = sp.load_npz(f'{DATA_DIR}/X.trn.npz')
X_test = sp.load_npz(f'{DATA_DIR}/X.tst.npz')

print(f"\nTraining set:")
print(f"  Documents: {Y_train.shape[0]:,}")
print(f"  Labels: {Y_train.shape[1]:,}")
print(f"  Features (TF-IDF): {X_train.shape[1]:,}")

print(f"\nTest set:")
print(f"  Documents: {Y_test.shape[0]:,}")
print(f"  Labels: {Y_test.shape[1]:,}")
print(f"  Features (TF-IDF): {X_test.shape[1]:,}")

# Data type and sparsity
print(f"\nData characteristics:")
print(f"  Y_train sparsity: {100 * (1 - Y_train.nnz / (Y_train.shape[0] * Y_train.shape[1])):.4f}%")
print(f"  X_train sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.4f}%")

# %% [markdown]
# ##### Explore labels

# %%
print("\n" + "="*80)
print("LABEL INFORMATION")
print("="*80)

# Load label descriptions
with open(f'{DATA_DIR}/label_map.txt', 'r', encoding='utf-8') as f:
    label_map = [line.strip() for line in f.readlines()]

print(f"\nTotal labels in label_map.txt: {len(label_map)}")

# Separate numeric and text labels
import re
numeric_labels = [(i, l) for i, l in enumerate(label_map) if re.match(r'^\d+\.?\d*$', l)]
text_labels = [(i, l) for i, l in enumerate(label_map) if not re.match(r'^\d+\.?\d*$', l)]

print(f"  Numeric labels (likely IDs): {len(numeric_labels)}")
print(f"  Text labels (descriptions): {len(text_labels)}")

print(f"\nFirst 20 labels:")
for i in range(20):
    print(f"  {i:4d}: {label_map[i]}")

print(f"\nLast 20 labels:")
for i in range(len(label_map)-20, len(label_map)):
    print(f"  {i:4d}: {label_map[i]}")

print(f"\nRandom sample of text labels:")
import random
sample_indices = random.sample([i for i, _ in text_labels], 20)
for idx in sorted(sample_indices):
    print(f"  {idx:4d}: {label_map[idx]}")

# %% [markdown]
# ##### Analyze label distribution

# %%
print("\n" + "="*80)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*80)

# Count how many times each label appears
label_counts_train = np.array(Y_train.sum(axis=0)).flatten()
label_counts_test = np.array(Y_test.sum(axis=0)).flatten()

print("\nLabel frequency statistics (Training set):")
print(f"  Min occurrences: {label_counts_train.min()}")
print(f"  Max occurrences: {label_counts_train.max()}")
print(f"  Mean occurrences: {label_counts_train.mean():.2f}")
print(f"  Median occurrences: {np.median(label_counts_train):.0f}")

# Head vs tail labels
print(f"\nLabel frequency distribution:")
tail_labels = np.sum(label_counts_train < 5)
print(f"  Tail labels (<5 occurrences): {tail_labels} ({100*tail_labels/len(label_counts_train):.1f}%)")
print(f"  Labels with 5-50 occurrences: {np.sum((label_counts_train >= 5) & (label_counts_train < 50))}")
print(f"  Labels with 50-500 occurrences: {np.sum((label_counts_train >= 50) & (label_counts_train < 500))}")
print(f"  Head labels (≥500 occurrences): {np.sum(label_counts_train >= 500)}")

# Top 20 most frequent labels
top_20_indices = np.argsort(label_counts_train)[-20:][::-1]
print(f"\nTop 20 most frequent labels (Training set):")
for rank, idx in enumerate(top_20_indices, 1):
    print(f"  {rank:2d}. Label {idx:4d} ({label_map[idx][:50]:50s}): {label_counts_train[idx]:5.0f} docs")

# Bottom 20 least frequent labels
bottom_20_indices = np.argsort(label_counts_train)[:20]
print(f"\nBottom 20 least frequent labels (Training set):")
for rank, idx in enumerate(bottom_20_indices, 1):
    if label_counts_train[idx] > 0:
        print(f"  {rank:2d}. Label {idx:4d} ({label_map[idx][:50]:50s}): {label_counts_train[idx]:5.0f} docs")

# %% [markdown]
# ##### Analyze Multi-Label Characteristics

# %%
print("\n" + "="*80)
print("MULTI-LABEL CHARACTERISTICS")
print("="*80)

# Labels per document
labels_per_doc_train = np.array(Y_train.sum(axis=1)).flatten()
labels_per_doc_test = np.array(Y_test.sum(axis=1)).flatten()

print("\nLabels per document (Training set):")
print(f"  Min: {labels_per_doc_train.min():.0f}")
print(f"  Max: {labels_per_doc_train.max():.0f}")
print(f"  Mean: {labels_per_doc_train.mean():.2f}")
print(f"  Median: {np.median(labels_per_doc_train):.0f}")

# Distribution
print(f"\nDistribution of labels per document (Training):")
label_dist = Counter(labels_per_doc_train.astype(int))
for num_labels in sorted(label_dist.keys())[:20]:  # First 20
    count = label_dist[num_labels]
    print(f"  {num_labels:2d} labels: {count:5d} documents ({100*count/Y_train.shape[0]:.1f}%)")

if len(label_dist) > 20:
    print(f"  ... ({len(label_dist) - 20} more categories)")

print(f"\nTest set:")
print(f"  Mean labels per doc: {labels_per_doc_test.mean():.2f}")
print(f"  Median labels per doc: {np.median(labels_per_doc_test):.0f}")

# %% [markdown]
# ##### Explore Raw Text

# %%
print("\n" + "="*80)
print("RAW TEXT EXPLORATION")
print("="*80)

# Load raw texts
with open(f'{DATA_DIR}/train_raw_texts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    train_texts = f.readlines()

with open(f'{DATA_DIR}/test_raw_texts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    test_texts = f.readlines()

print(f"\nNumber of training documents: {len(train_texts)}")
print(f"Number of test documents: {len(test_texts)}")

# Text length analysis
train_lengths = [len(text.split()) for text in train_texts]

print(f"\nText length statistics (in words):")
print(f"  Min: {min(train_lengths)}")
print(f"  Max: {max(train_lengths)}")
print(f"  Mean: {np.mean(train_lengths):.1f}")
print(f"  Median: {np.median(train_lengths):.0f}")

# Length distribution
print(f"\nText length distribution:")
bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]
for i in range(len(bins)-1):
    count = sum(1 for l in train_lengths if bins[i] <= l < bins[i+1])
    print(f"  {bins[i]:5d}-{bins[i+1]:5d} words: {count:5d} docs ({100*count/len(train_lengths):.1f}%)")
count = sum(1 for l in train_lengths if l >= bins[-1])
print(f"  {bins[-1]:5d}+ words: {count:5d} docs ({100*count/len(train_lengths):.1f}%)")

print(f"\n" + "-"*80)
print("SAMPLE TRAINING DOCUMENTS:")
print("-"*80)

# Show 3 random documents
for i in random.sample(range(len(train_texts)), 3):
    print(f"\nDocument {i}:")
    print(f"  Labels: {np.where(Y_train[i].toarray()[0] == 1)[0]}")
    label_names = [label_map[idx] for idx in np.where(Y_train[i].toarray()[0] == 1)[0]]
    print(f"  Label names: {label_names}")
    print(f"  Text (first 300 chars):")
    print(f"    {train_texts[i][:300]}...")

# %% [markdown]
# ##### Label Co-occurrence Analysis
# 

# %%
print("\n" + "="*80)
print("LABEL CO-OCCURRENCE ANALYSIS")
print("="*80)

# Find most common label pairs
from itertools import combinations

print("\nAnalyzing label co-occurrences (this may take a moment)...")

label_pairs = Counter()
for i in range(min(1000, Y_train.shape[0])):  # Sample first 1000 docs
    doc_labels = np.where(Y_train[i].toarray()[0] == 1)[0]
    if len(doc_labels) >= 2:
        for pair in combinations(doc_labels, 2):
            label_pairs[tuple(sorted(pair))] += 1

print(f"\nTop 20 most common label pairs (in first 1000 documents):")
for rank, (pair, count) in enumerate(label_pairs.most_common(20), 1):
    l1, l2 = pair
    name1 = label_map[l1][:30]
    name2 = label_map[l2][:30]
    print(f"  {rank:2d}. ({l1:4d}, {l2:4d}) = ({name1:30s}, {name2:30s}): {count:3d} times")

# %% [markdown]
# ##### Dataset Summary

# %%
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

print(f"""
EURLEX-4K Dataset Characteristics:

Documents:
  - Training: {Y_train.shape[0]:,}
  - Test: {Y_test.shape[0]:,}
  - Avg text length: {np.mean(train_lengths):.0f} words

Labels:
  - Total labels: {Y_train.shape[1]:,}
  - Text labels: {len(text_labels):,}
  - Numeric labels: {len(numeric_labels)}
  - Avg labels per doc: {labels_per_doc_train.mean():.2f}
  - Tail labels (<5 docs): {tail_labels} ({100*tail_labels/len(label_counts_train):.1f}%)

Features:
  - TF-IDF vocabulary size: {X_train.shape[1]:,}
  - Feature matrix sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%

Label Language:
  - English text descriptions: {len(text_labels):,}
  - Need translation for multilingual experiments
""")

# %%



