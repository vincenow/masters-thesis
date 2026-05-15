"""
Supervised baseline: XLM-RoBERTa-large fine-tuned on MultiEURLEX (English only),
evaluated on EN, FR, NL, DE test sets (zero-shot cross-lingual transfer).

Output: results_xlmr_{language}_supervised.json — same format as retrieval results.

Usage:
    python supervised_baseline.py --output_dir ./results
    python supervised_baseline.py --output_dir ./results --skip_training --model_path ./xlmr_checkpoint

Requirements:
    pip install transformers datasets scikit-learn torch accelerate
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "xlm-roberta-large"
LANGUAGES  = ["en", "fr", "nl", "de"]
TRAIN_LANG = "en"

MAX_LENGTH   = 512
BATCH_SIZE   = 8          # per GPU; reduce to 4 if OOM
GRAD_ACCUM   = 4          # effective batch = 32
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
THRESHOLD    = 0.5        # sigmoid threshold for binary predictions
K_VALUES     = [5, 10, 20, 50, 100]

LABEL_LEVEL  = "all_levels"   # 7390 labels


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiEURLEXDataset(Dataset):
    def __init__(self, hf_split, language, tokenizer, mlb, max_length=512):
        self.texts      = [ex["text"][language] for ex in hf_split]
        self.labels     = mlb.transform([ex["labels"] for ex in hf_split])
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class XLMRClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens
        token_embeddings = outputs.last_hidden_state          # (B, T, H)
        mask_expanded    = attention_mask.unsqueeze(-1).float()
        pooled           = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        logits           = self.classifier(pooled)            # (B, num_labels)
        return logits


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(scores, true_labels, k):
    """scores: (N, L) numpy, true_labels: (N, L) binary numpy"""
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]
    hits = np.array([true_labels[i, top_k_idx[i]].sum() for i in range(len(scores))])
    return hits / k

def recall_at_k(scores, true_labels, k):
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]
    hits      = np.array([true_labels[i, top_k_idx[i]].sum() for i in range(len(scores))])
    totals    = true_labels.sum(axis=1).clip(min=1)
    return hits / totals

def ndcg_at_k(scores, true_labels, k):
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]
    ndcgs = []
    for i in range(len(scores)):
        rel   = true_labels[i, top_k_idx[i]]          # relevance of top-k
        gains = rel / np.log2(np.arange(2, k + 2))    # discounted gains
        dcg   = gains.sum()
        # Ideal DCG
        n_rel = int(true_labels[i].sum())
        ideal_k = min(n_rel, k)
        if ideal_k == 0:
            ndcgs.append(0.0)
            continue
        ideal_gains = np.ones(ideal_k) / np.log2(np.arange(2, ideal_k + 2))
        idcg        = ideal_gains.sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(ndcgs)

def compute_metrics(scores, true_labels, k_values):
    """Returns dict of {k: {P, R, NDCG, P_std, R_std, NDCG_std}}"""
    # Only evaluate on documents that have at least one true label
    mask = true_labels.sum(axis=1) > 0
    scores      = scores[mask]
    true_labels = true_labels[mask]

    results = {}
    for k in k_values:
        p    = precision_at_k(scores, true_labels, k)
        r    = recall_at_k(scores, true_labels, k)
        ndcg = ndcg_at_k(scores, true_labels, k)
        results[k] = {
            "P@k":      float(p.mean()),
            "R@k":      float(r.mean()),
            "NDCG@k":   float(ndcg.mean()),
            "P@k_std":  float(p.std()),
            "R@k_std":  float(r.std()),
            "NDCG@k_std": float(ndcg.std()),
        }
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, device, epochs, lr, warmup_ratio, grad_accum):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    for batch in eval_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits         = model(input_ids, attention_mask)
        scores         = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(batch["labels"].numpy())
    return np.vstack(all_scores), np.vstack(all_labels)


# ── Output format (matches retrieval results) ─────────────────────────────────

def build_output(language, metrics_by_k, n_docs):
    """
    Mirrors the JSON structure of results_{model}_{language}_{labelcondition}.json.
    Adjust keys here if your retrieval JSON uses different field names.
    """
    rows = []
    for k, m in metrics_by_k.items():
        rows.append({
            "k":       k,
            "P@k":     round(m["P@k"],    4),
            "R@k":     round(m["R@k"],    4),
            "NDCG@k":  round(m["NDCG@k"], 4),
            "P@k_std":    round(m["P@k_std"],    4),
            "R@k_std":    round(m["R@k_std"],    4),
            "NDCG@k_std": round(m["NDCG@k_std"], 4),
        })
    return {
        "model":          "xlm-roberta-large",
        "language":       language,
        "label_language": "supervised",
        "n_docs":         n_docs,
        "results":        rows,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",    type=str, default="./results")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load from --model_path")
    parser.add_argument("--model_path",    type=str, default="./xlmr_checkpoint",
                        help="Path to save/load fine-tuned model")
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",        type=int, default=EPOCHS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading MultiEURLEX …")
    dataset = load_dataset(
        "coastalcph/multi_eurlex",
        "all_languages",
        label_level=LABEL_LEVEL,
        trust_remote_code=True,
    )

    # Build MLB from training split (all label IDs, not just those seen in EN)
    all_label_ids = [str(l) for ex in dataset["train"] for l in ex["labels"]]
    unique_labels = sorted(set(all_label_ids))
    mlb = MultiLabelBinarizer(classes=unique_labels)
    mlb.fit([[l] for l in unique_labels])
    num_labels = len(unique_labels)
    print(f"Number of labels: {num_labels}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not args.skip_training:
        print(f"Building training set ({TRAIN_LANG}) …")
        train_ds = MultiEURLEXDataset(
            dataset["train"], TRAIN_LANG, tokenizer, mlb, MAX_LENGTH
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        print(f"Fine-tuning {MODEL_NAME} …")
        model = XLMRClassifier(MODEL_NAME, num_labels).to(device)

        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model = train(model, train_loader, device, args.epochs, LR, WARMUP_RATIO, GRAD_ACCUM)

        # Save
        core = model.module if hasattr(model, "module") else model
        torch.save(core.state_dict(), os.path.join(args.model_path, "model.pt"))
        tokenizer.save_pretrained(args.model_path)
        with open(os.path.join(args.model_path, "mlb_classes.json"), "w") as f:
            json.dump(unique_labels, f)
        print(f"Model saved to {args.model_path}")
    else:
        print(f"Loading model from {args.model_path} …")
        with open(os.path.join(args.model_path, "mlb_classes.json")) as f:
            unique_labels = json.load(f)
        mlb = MultiLabelBinarizer(classes=unique_labels)
        mlb.fit([[l] for l in unique_labels])
        num_labels = len(unique_labels)

        model = XLMRClassifier(MODEL_NAME, num_labels)
        model.load_state_dict(
            torch.load(os.path.join(args.model_path, "model.pt"), map_location=device)
        )
        model = model.to(device)

    # ── Evaluate per language ─────────────────────────────────────────────────
    for lang in LANGUAGES:
        print(f"\nEvaluating on {lang} test set …")
        test_ds = MultiEURLEXDataset(
            dataset["test"], lang, tokenizer, mlb, MAX_LENGTH
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        scores, true_labels = evaluate(model, test_loader, device)
        metrics = compute_metrics(scores, true_labels, K_VALUES)

        output = build_output(lang, metrics, n_docs=len(test_ds))
        out_path = os.path.join(args.output_dir, f"results_xlmr_{lang}_supervised.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved → {out_path}")

        # Quick summary
        for k in K_VALUES:
            m = metrics[k]
            print(f"  k={k:3d}  P@k={m['P@k']:.4f}  R@k={m['R@k']:.4f}  NDCG@k={m['NDCG@k']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()