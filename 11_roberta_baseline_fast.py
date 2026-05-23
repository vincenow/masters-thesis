"""
Supervised baseline: XLM-RoBERTa-large fine-tuned on MultiEURLEX (English only),
evaluated on EN, FR, NL, DE test sets (zero-shot cross-lingual transfer).

Output: results_xlmr_{language}_supervised.json — same format as retrieval results.

Usage:
    python 11_roberta_baseline.py --output_dir ./results --epochs 3
    python 11_roberta_baseline.py --output_dir ./results --skip_training --model_path ./xlmr_checkpoint/epoch_1
    python 11_roberta_baseline.py --output_dir ./results --resume_from_epoch 1 --epochs 3

Requirements:
    pip install transformers datasets scikit-learn torch accelerate
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "xlm-roberta-large"
LANGUAGES    = ["en", "fr", "nl", "de"]
TRAIN_LANG   = "en"
MAX_LENGTH   = 512
BATCH_SIZE   = 16         # larger batch with fp16
GRAD_ACCUM   = 2          # effective batch = 32
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
K_VALUES     = [5, 10, 20, 50, 100]
LABEL_LEVEL  = "all_levels"


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiEURLEXDataset(Dataset):
    def __init__(self, hf_split, language, tokenizer, mlb, max_length=512):
        texts = [ex["text"][language] for ex in hf_split]
        self.labels = mlb.transform([ex["labels"] for ex in hf_split])
        print(f"  Tokenizing {len(texts)} documents …", flush=True)
        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
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


# ── Model ─────────────────────────────────────────────────────────────────────

class XLMRClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs          = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded    = attention_mask.unsqueeze(-1).float()
        pooled           = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(scores, true_labels, k):
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
        rel     = true_labels[i, top_k_idx[i]]
        gains   = rel / np.log2(np.arange(2, k + 2))
        dcg     = gains.sum()
        n_rel   = int(true_labels[i].sum())
        ideal_k = min(n_rel, k)
        if ideal_k == 0:
            ndcgs.append(0.0)
            continue
        ideal_gains = np.ones(ideal_k) / np.log2(np.arange(2, ideal_k + 2))
        idcg        = ideal_gains.sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(ndcgs)

def compute_metrics(scores, true_labels, k_values):
    mask        = true_labels.sum(axis=1) > 0
    scores      = scores[mask]
    true_labels = true_labels[mask]
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

def save_checkpoint(model, tokenizer, unique_labels, model_path, epoch):
    ckpt_dir = os.path.join(model_path, f"epoch_{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    torch.save(core.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "mlb_classes.json"), "w") as f:
        json.dump(unique_labels, f)
    print(f"  Checkpoint saved → {ckpt_dir}", flush=True)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, device, epochs, lr, warmup_ratio, grad_accum,
          tokenizer, unique_labels, model_path, start_epoch=0):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.BCEWithLogitsLoss()
    scaler       = GradScaler()  # fp16 scaler

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast():  # fp16 forward pass
                logits = model(input_ids, attention_mask)
                loss   = loss_fn(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} step {step}/{len(train_loader)} — loss: {loss.item() * grad_accum:.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} complete — avg loss: {avg_loss:.4f}", flush=True)
        save_checkpoint(model, tokenizer, unique_labels, model_path, epoch + 1)

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for batch in eval_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.vstack(all_scores), np.vstack(all_labels)


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
    parser.add_argument("--output_dir",        type=str, default="./results")
    parser.add_argument("--skip_training",     action="store_true")
    parser.add_argument("--model_path",        type=str, default="./xlmr_checkpoint")
    parser.add_argument("--batch_size",        type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",            type=int, default=EPOCHS)
    parser.add_argument("--resume_from_epoch", type=int, default=0,
                        help="Resume from epoch N checkpoint (loads from model_path/epoch_N)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading MultiEURLEX …", flush=True)
    dataset = load_dataset(
        "coastalcph/multi_eurlex",
        "all_languages",
        label_level=LABEL_LEVEL,
        trust_remote_code=True,
    )

    all_label_ids = [str(l) for ex in dataset["train"] for l in ex["labels"]]
    unique_labels = sorted(set(all_label_ids))
    mlb = MultiLabelBinarizer(classes=unique_labels)
    mlb.fit([[l] for l in unique_labels])
    num_labels = len(unique_labels)
    print(f"Number of labels: {num_labels}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not args.skip_training:
        if args.resume_from_epoch > 0:
            ckpt_dir = os.path.join(args.model_path, f"epoch_{args.resume_from_epoch}")
            print(f"Resuming from checkpoint: {ckpt_dir}", flush=True)
            with open(os.path.join(ckpt_dir, "mlb_classes.json")) as f:
                unique_labels = json.load(f)
            mlb = MultiLabelBinarizer(classes=unique_labels)
            mlb.fit([[l] for l in unique_labels])
            num_labels = len(unique_labels)
            model = XLMRClassifier(MODEL_NAME, num_labels)
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device))
            model = model.to(device)
        else:
            model = XLMRClassifier(MODEL_NAME, num_labels).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
            model = nn.DataParallel(model)

        print(f"Building training set ({TRAIN_LANG}) …", flush=True)
        train_ds = MultiEURLEXDataset(
            dataset["train"], TRAIN_LANG, tokenizer, mlb, MAX_LENGTH
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

        print(f"Fine-tuning {MODEL_NAME} with fp16 …", flush=True)
        model = train(
            model, train_loader, device, args.epochs, LR, WARMUP_RATIO, GRAD_ACCUM,
            tokenizer, unique_labels, args.model_path,
            start_epoch=args.resume_from_epoch,
        )

        core = model.module if hasattr(model, "module") else model
        torch.save(core.state_dict(), os.path.join(args.model_path, "model.pt"))
        tokenizer.save_pretrained(args.model_path)
        with open(os.path.join(args.model_path, "mlb_classes.json"), "w") as f:
            json.dump(unique_labels, f)
        print(f"Final model saved to {args.model_path}", flush=True)

    else:
        print(f"Loading model from {args.model_path} …", flush=True)
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
        print(f"\nEvaluating on {lang} test set …", flush=True)
        test_ds = MultiEURLEXDataset(
            dataset["test"], lang, tokenizer, mlb, MAX_LENGTH
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        scores, true_labels = evaluate(model, test_loader, device)
        metrics = compute_metrics(scores, true_labels, K_VALUES)

        output = build_output(lang, metrics, n_docs=len(test_ds))
        out_path = os.path.join(args.output_dir, f"results_xlmr_{lang}_supervised.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved → {out_path}", flush=True)

        for k in K_VALUES:
            m = metrics[k]
            print(f"  k={k:3d}  P@k={m['P@k']:.4f}  R@k={m['R@k']:.4f}  NDCG@k={m['NDCG@k']:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()