"""
Supervised baseline: XLM-RoBERTa-large fine-tuned on MultiEURLEX (English only),
evaluated on EN, FR, NL, DE test sets (zero-shot cross-lingual transfer).

Closely follows Chalkidis et al. (2021):
- CLS token + dropout(0.1) before classifier
- Classifier kernel init: TruncatedNormal(std=0.02)
- ALL labels from train+dev+test splits (4591), not just train (4220)
- LR = 3e-5, batch_size = 8
- Up to 70 epochs, early stopping patience 5 on dev mRP
- No label smoothing
- Plain BCEWithLogitsLoss

Output: results_xlmr_{language}_supervised.json
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "xlm-roberta-large"
LANGUAGES    = ["en", "fr", "nl", "de"]
TRAIN_LANG   = "en"
MAX_LENGTH   = 512
BATCH_SIZE   = 8
GRAD_ACCUM   = 1
MAX_EPOCHS   = 70
LR           = 3e-5
WARMUP_RATIO = 0.1
PATIENCE     = 5
K_VALUES     = [5, 10, 20, 50, 100]
LABEL_LEVEL  = "all_levels"


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiEURLEXDataset(Dataset):
    def __init__(self, hf_split, language, tokenizer, mlb, max_length=512):
        texts = [ex["text"][language] for ex in hf_split]
        self.labels = mlb.transform([ex["labels"] for ex in hf_split])
        print(f"  Tokenizing {len(texts)} documents …", flush=True)
        enc = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Model (matches Chalkidis et al. 2021) ─────────────────────────────────────

class XLMRClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden_size   = self.encoder.config.hidden_size
        self.dropout  = nn.Dropout(0.1)
        # TruncatedNormal(std=0.02) matches Chalkidis et al.
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # CLS token
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


# ── Metrics ───────────────────────────────────────────────────────────────────

def mean_r_precision(scores, true_labels):
    rp_scores = []
    for i in range(len(scores)):
        n_true = int(true_labels[i].sum())
        if n_true == 0:
            continue
        top_k = np.argsort(-scores[i])[:n_true]
        rp_scores.append(true_labels[i, top_k].sum() / n_true)
    return float(np.mean(rp_scores)) if rp_scores else 0.0

def precision_at_k(scores, true_labels, k):
    top_k = np.argsort(-scores, axis=1)[:, :k]
    hits  = np.array([true_labels[i, top_k[i]].sum() for i in range(len(scores))])
    return hits / k

def recall_at_k(scores, true_labels, k):
    top_k  = np.argsort(-scores, axis=1)[:, :k]
    hits   = np.array([true_labels[i, top_k[i]].sum() for i in range(len(scores))])
    totals = true_labels.sum(axis=1).clip(min=1)
    return hits / totals

def ndcg_at_k(scores, true_labels, k):
    top_k = np.argsort(-scores, axis=1)[:, :k]
    ndcgs = []
    for i in range(len(scores)):
        rel     = true_labels[i, top_k[i]]
        gains   = rel / np.log2(np.arange(2, k + 2))
        dcg     = gains.sum()
        n_rel   = int(true_labels[i].sum())
        ideal_k = min(n_rel, k)
        if ideal_k == 0:
            ndcgs.append(0.0)
            continue
        idcg = (np.ones(ideal_k) / np.log2(np.arange(2, ideal_k + 2))).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(ndcgs)

def compute_metrics(scores, true_labels, k_values):
    results = {}
    for k in k_values:
        p    = precision_at_k(scores, true_labels, k)
        r    = recall_at_k(scores, true_labels, k)
        ndcg = ndcg_at_k(scores, true_labels, k)
        results[k] = {
            "P@k":        float(p.mean()),
            "R@k":        float(r.mean()),
            "NDCG@k":     float(ndcg.mean()),
            "P@k_std":    float(p.std()),
            "R@k_std":    float(r.std()),
            "NDCG@k_std": float(ndcg.std()),
        }
    return results


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, unique_labels, model_path, tag):
    ckpt_dir = os.path.join(model_path, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    torch.save({k: v.cpu() for k, v in core.state_dict().items()},
               os.path.join(ckpt_dir, "model.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "mlb_classes.json"), "w") as f:
        json.dump(unique_labels, f)
    print(f"  Checkpoint saved → {ckpt_dir}", flush=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.vstack(all_scores), np.vstack(all_labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, dev_loader, device, max_epochs, lr,
          warmup_ratio, patience, tokenizer, unique_labels, model_path):

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps  = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.BCEWithLogitsLoss()

    best_mrp       = -1.0
    best_epoch     = -1
    patience_count = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels)
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 500 == 0:
                print(f"  Epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"— loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)

        # Dev mRP for early stopping
        dev_scores, dev_labels = run_eval(model, dev_loader, device)
        mrp = mean_r_precision(dev_scores, dev_labels)
        print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}  dev mRP: {mrp:.4f}", flush=True)

        # Save per-epoch checkpoint
        save_checkpoint(model, tokenizer, unique_labels, model_path, f"epoch_{epoch+1}")

        if mrp > best_mrp:
            best_mrp       = mrp
            best_epoch     = epoch + 1
            patience_count = 0
            core = model.module if hasattr(model, "module") else model
            torch.save({k: v.cpu() for k, v in core.state_dict().items()},
                       os.path.join(model_path, "model_best.pt"))
            print(f"  ★ New best model (mRP={best_mrp:.4f}) saved.", flush=True)
        else:
            patience_count += 1
            print(f"  No improvement. Patience: {patience_count}/{patience}", flush=True)
            if patience_count >= patience:
                print(f"Early stopping. Best epoch: {best_epoch}, mRP: {best_mrp:.4f}", flush=True)
                break

    return best_epoch


# ── Output format ─────────────────────────────────────────────────────────────

def build_output(language, metrics_by_k, n_docs):
    rows = []
    for k, m in metrics_by_k.items():
        rows.append({
            "k":          k,
            "P@k":        round(m["P@k"],        4),
            "R@k":        round(m["R@k"],        4),
            "NDCG@k":     round(m["NDCG@k"],     4),
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
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--model_path",    type=str, default="./xlmr_checkpoint")
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs",    type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading MultiEURLEX …", flush=True)
    dataset = load_dataset(
        "coastalcph/multi_eurlex", "all_languages",
        label_level=LABEL_LEVEL, trust_remote_code=True,
    )

    # Build MLB from ALL splits (train+dev+test) = 4591 labels
    # This matches Chalkidis et al. who used eurovoc_concepts.json with all labels
    print("Building label index from all splits …", flush=True)
    all_label_ids = [
        str(l)
        for split in ["train", "validation", "test"]
        for ex in dataset[split]
        for l in ex["labels"]
    ]
    unique_labels = sorted(set(all_label_ids))
    mlb = MultiLabelBinarizer(classes=unique_labels)
    mlb.fit([[l] for l in unique_labels])
    num_labels = len(unique_labels)
    print(f"Number of labels: {num_labels}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not args.skip_training:
        print(f"Building training set ({TRAIN_LANG}) …", flush=True)
        train_ds = MultiEURLEXDataset(dataset["train"],      TRAIN_LANG, tokenizer, mlb, MAX_LENGTH)
        dev_ds   = MultiEURLEXDataset(dataset["validation"], TRAIN_LANG, tokenizer, mlb, MAX_LENGTH)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size * 4, shuffle=False,
                                  num_workers=0, pin_memory=True)

        print(f"Fine-tuning {MODEL_NAME} …", flush=True)
        model = XLMRClassifier(MODEL_NAME, num_labels).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
            model = nn.DataParallel(model)

        best_epoch = train(
            model, train_loader, dev_loader, device,
            args.max_epochs, LR, WARMUP_RATIO, PATIENCE,
            tokenizer, unique_labels, args.model_path,
        )

        tokenizer.save_pretrained(args.model_path)
        with open(os.path.join(args.model_path, "mlb_classes.json"), "w") as f:
            json.dump(unique_labels, f)
        print(f"Best model from epoch {best_epoch} saved.", flush=True)

        # Load best model for evaluation
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(
            torch.load(os.path.join(args.model_path, "model_best.pt"), map_location=device)
        )

    else:
        print(f"Loading model from {args.model_path} …", flush=True)
        with open(os.path.join(args.model_path, "mlb_classes.json")) as f:
            unique_labels = json.load(f)
        mlb = MultiLabelBinarizer(classes=unique_labels)
        mlb.fit([[l] for l in unique_labels])
        num_labels = len(unique_labels)
        model = XLMRClassifier(MODEL_NAME, num_labels)
        model.load_state_dict(
            torch.load(os.path.join(args.model_path, "model_best.pt"), map_location="cpu")
        )
        model = model.to(device)

    # ── Evaluate per language ─────────────────────────────────────────────────
    for lang in LANGUAGES:
        print(f"\nEvaluating on {lang} test set …", flush=True)
        test_ds = MultiEURLEXDataset(dataset["test"], lang, tokenizer, mlb, MAX_LENGTH)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, shuffle=False,
                                 num_workers=0, pin_memory=True)

        scores, true_labels = run_eval(model, test_loader, device)
        metrics  = compute_metrics(scores, true_labels, K_VALUES)
        output   = build_output(lang, metrics, n_docs=len(test_ds))
        out_path = os.path.join(args.output_dir, f"results_xlmr_{lang}_supervised.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved → {out_path}", flush=True)

        mrp = mean_r_precision(scores, true_labels)
        print(f"  mRP: {mrp:.4f}", flush=True)
        for k in K_VALUES:
            m = metrics[k]
            print(f"  k={k:3d}  P@k={m['P@k']:.4f}  R@k={m['R@k']:.4f}  NDCG@k={m['NDCG@k']:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()