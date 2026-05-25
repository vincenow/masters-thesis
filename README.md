# Multilingual Extreme Multi-Label Classification without Task-Specific Training
## A Zero-Shot Retrieval Study on Legal Documents

**Vincent Hagenow**  
University of Antwerp, 2026

---

## Overview

This repository contains the code, results, and figures for my master's thesis, which investigates whether pre-trained multilingual embedding models can perform zero-shot label retrieval for Extreme Multi-Label Text Classification (XMTC) on legal documents without any task-specific training.

The approach treats label assignment as a retrieval problem: documents and EuroVoc label descriptors are embedded using a pre-trained model, and the top-k most similar labels are retrieved via cosine similarity. An optional cross-encoder reranking stage is applied on top of the top-100 candidates. Three alternative label representation strategies are also evaluated: native-language descriptors, AI-generated descriptors, and centroid vectors built from training documents.

**Research questions:**
- **RQ1:** Which embedding models perform best for zero-shot label retrieval in the multilingual legal domain?
- **RQ2a:** Does using label descriptors in the same language as the query document improve retrieval accuracy over always using English labels?
- **RQ2b:** Does replacing the short official EuroVoc descriptor phrases with richer, AI-generated descriptions improve retrieval accuracy?
- **RQ2c:** Does representing labels as centroid vectors of training documents improve retrieval accuracy over text-based label representations?

**Dataset:** [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex) — multilingual EU legislative documents annotated with EuroVoc labels. Of the 7,390 concepts in the EuroVoc thesaurus, 4,591 are actually assigned to at least one document in the dataset. Experiments are conducted on the test set across four languages: English, French, Dutch, and German.

**Embedding models evaluated:** multilingual-E5-small, LaBSE, OpenAI text-embedding-3-small, gte-multilingual-base, BGE-M3, Qwen3-Embedding-0.6B, harrier-oss-v1

**Reranker:** BGE-reranker-v2-m3 (cross-encoder, applied to top-100 candidates)

**Supervised reference:** XLM-RoBERTa-large fine-tuned on the English training split, evaluated zero-shot on all four languages

**Metrics:** Precision@k, Recall@k, NDCG@k at k = 5, 10, 20, 50, 100

---

## Repository Structure

```
masters-thesis/
├── data/               EuroVoc descriptor files and generated label descriptions
├── embeddings/         Precomputed .npy embedding files (see note below)
├── results/            Evaluation results as JSON files, one per model/language/condition
├── figures/            All plots and figures used in the thesis
├── notebooks/          Jupyter notebooks for exploration and early model experiments
├── scripts/            Python scripts for model evaluation, reranking, and figure generation
├── requirements.txt
└── requirements_no_torch.txt
```

### Key files

| File | Description |
|------|-------------|
| `data/eurovoc_descriptors.json` | EuroVoc label descriptors in all languages |
| `data/eurovoc_scope_notes.json` | EuroVoc scope note enrichments |
| `data/generated_descriptors.json` | AI-generated label descriptions (RQ2b) |
| `notebooks/100_load_all_results.ipynb` | Main analysis notebook — loads and compares all results |
| `scripts/0_figures_results_*.py` | Figure generation scripts |

### Result file naming convention

```
results_{model}_{language}_{label_condition}.json

model:           e5, labse, openai, gte, bge_m3, qwen3, harrier, bm25
language:        en, fr, nl, de
label_condition: en_labels, native_labels, generated_labels,
                 reranked_en_labels, reranked_native_labels
```

---

## Environment

Experiments were run on a remote Ubuntu server with 8× NVIDIA RTX 2080 Ti GPUs (11 GB VRAM each).

- **Python:** 3.13.13
- **Environment manager:** Miniconda (`thesis_env`)

### Setup

```bash
conda create -n thesis_env python=3.13
conda activate thesis_env
pip install -r requirements.txt
```

### Key dependencies

| Package | Version |
|---------|---------|
| sentence-transformers | 5.5.0 |
| FlagEmbedding | 1.4.0 |
| transformers | 5.8.1 |
| torch | 2.12.0 |
| datasets | 2.21.0 |
| openai | 2.37.0 |
| scikit-learn | 1.7.2 |

---

## Large Files Not Tracked in Git

The following are not committed to this repository:

- Model weights (stored separately on remote server)

OpenAI embeddings require a valid API key set as `OPENAI_API_KEY` in a `.env` file.
