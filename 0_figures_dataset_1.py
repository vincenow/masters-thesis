"""
visualize_dataset.py
────────────────────
Generates descriptive statistics and figures for the MultiEURLEX dataset
chapter of the thesis. Run once; all figures are saved to ./figures/.

Font: Poppins (auto-installed via pip if absent), with fallback chain:
      Liberation Sans → DejaVu Sans → sans-serif. The script prints which
      font is actually used.

Produces:
  1. figures/table_dataset_stats.png   — booktabs-style split stats table
  2. figures/label_frequency_bins.png  — binned label-frequency bar chart
  3. figures/labels_per_doc.png        — distribution of labels per document
"""

import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from datasets import load_dataset

# ── tokenizers for token count stats ─────────────────────────────────────────
# BERT-style: bert-base-multilingual-cased (512 token limit)
# GPT-style:  cl100k_base via tiktoken (OpenAI, 8192 token limit)
try:
    from transformers import AutoTokenizer
    try:
        _bert_tok = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=True)
    except Exception:
        _bert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    HAVE_BERT_TOK = True
    print("BERT tokenizer loaded.")
except Exception as e:
    HAVE_BERT_TOK = False
    print(f"Warning: BERT tokenizer failed ({e}); BERT token counts skipped.")

try:
    import tiktoken
    _gpt_tok = tiktoken.get_encoding("cl100k_base")
    HAVE_GPT_TOK = True
    print("GPT tokenizer loaded.")
except Exception as e:
    HAVE_GPT_TOK = False
    print(f"Warning: tiktoken failed ({e}); GPT token counts skipped.")

# ── output directory ──────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ── palette & global style ────────────────────────────────────────────────────
C = {
    "blue":   "#2E5FA3",   # deep navy blue
    "red":    "#B03A3A",   # muted crimson
    "yellow": "#C9A84C",   # warm antique gold
    "grey":   "#888888",
    "lgrey":  "#DADADA",
    "ink":    "#1A1A1A",
    "bg":     "#FFFFFF",
}

# ── font: DejaVu Sans is matplotlib's own bundled font — always available ─────
# Clean, neutral, no system font cache issues. Swap FONT to any installed name.
FONT = "DejaVu Sans"

rcParams.update({
    "font.family":       FONT,
    "font.size":         9.5,
    "text.color":        C["ink"],
    "axes.labelcolor":   C["ink"],
    "xtick.color":       C["ink"],
    "ytick.color":       C["ink"],
    "figure.facecolor":  C["bg"],
    "axes.facecolor":    C["bg"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "axes.edgecolor":    C["lgrey"],
    "axes.grid":         True,
    "grid.color":        "#EFEFEF",
    "grid.linewidth":    0.7,
    "axes.axisbelow":    True,
    "legend.frameon":    False,
    "legend.fontsize":   9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading splits (English; labels are language-agnostic)…")
splits = {}
for split in ("train", "validation", "test"):
    splits[split] = load_dataset(
        "coastalcph/multi_eurlex", "en",
        split=split,
        label_level="all_levels",
        trust_remote_code=True,
    )
    print(f"  {split}: {len(splits[split]):,}")

classlabel   = splits["train"].features["labels"].feature
TOTAL_LABELS = len(classlabel.names)   # 7,390

# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def split_stats(dataset):
    label_counts    = collections.Counter()
    labels_per_doc  = []
    word_counts     = []
    bert_tok_counts = []
    gpt_tok_counts  = []
    for doc in dataset:
        labs = doc["labels"]
        labels_per_doc.append(len(labs))
        label_counts.update(labs)
        text = doc["text"] or ""
        word_counts.append(len(text.split()))
        if HAVE_BERT_TOK:
            # add_special_tokens=False to count raw content tokens
            bert_tok_counts.append(len(_bert_tok.encode(text, add_special_tokens=False)))
        if HAVE_GPT_TOK:
            gpt_tok_counts.append(len(_gpt_tok.encode(text)))
    n = len(dataset)
    u = len(label_counts)
    return {
        "n_docs":               n,
        "unique_labels":        u,
        "avg_labels_per_doc":   np.mean(labels_per_doc),
        "median_labels_per_doc":np.median(labels_per_doc),
        "avg_docs_per_label":   n * np.mean(labels_per_doc) / u,
        "avg_words":            np.mean(word_counts),
        "avg_bert_tokens":      np.mean(bert_tok_counts) if bert_tok_counts else None,
        "avg_gpt_tokens":       np.mean(gpt_tok_counts)  if gpt_tok_counts  else None,
        "labels_per_doc_arr":   labels_per_doc,
        "label_counts":         label_counts,
    }

print("\nComputing statistics…")
stats = {s: split_stats(splits[s]) for s in ("train", "validation", "test")}
train_lc = stats["train"]["label_counts"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — DATASET STATISTICS TABLE  (booktabs-style)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 1: dataset stats table…")

tr = stats["train"]
va = stats["validation"]
te = stats["test"]

def _fmt(val, fmt=".0f"):
    return f"{val:{fmt}}" if val is not None else "n/a"

rows = [
    # (Statistic,                  Train,              Val,              Test)
    ("Documents",
        f"{tr['n_docs']:,}",       f"{va['n_docs']:,}", f"{te['n_docs']:,}"),
    ("Unique EuroVoc labels",
        f"{tr['unique_labels']:,}",f"{va['unique_labels']:,}",f"{te['unique_labels']:,}"),
    ("Avg labels / document",
        f"{tr['avg_labels_per_doc']:.2f}",
        f"{va['avg_labels_per_doc']:.2f}",
        f"{te['avg_labels_per_doc']:.2f}"),
    ("Median labels / document",
        f"{tr['median_labels_per_doc']:.0f}",
        f"{va['median_labels_per_doc']:.0f}",
        f"{te['median_labels_per_doc']:.0f}"),
    ("Avg documents / label",
        f"{tr['avg_docs_per_label']:.1f}", "—", "—"),
    ("Avg words / document",
        f"{tr['avg_words']:.0f}",
        f"{va['avg_words']:.0f}",
        f"{te['avg_words']:.0f}"),
    ("Avg BERT tokens / document",
        _fmt(tr['avg_bert_tokens']),
        _fmt(va['avg_bert_tokens']),
        _fmt(te['avg_bert_tokens'])),
    ("Avg GPT tokens / document",
        _fmt(tr['avg_gpt_tokens']),
        _fmt(va['avg_gpt_tokens']),
        _fmt(te['avg_gpt_tokens'])),
]

col_headers = ["", "Train", "Validation", "Test"]
n_rows = len(rows)
n_cols = 4

fig_h = 0.42 + n_rows * 0.32
fig, ax = plt.subplots(figsize=(6.8, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# column x-positions (left edges used for text anchors)
col_x   = [0.02, 0.52, 0.70, 0.86]
col_ha  = ["left", "right", "right", "right"]
col_x_right = [0.50, 0.68, 0.84, 1.0]   # right boundary per col (for alignment reference)

row_h   = 1.0 / (n_rows + 2.5)           # +2.5 leaves room for header + rules
header_y = 1.0 - row_h * 0.55

# ── top rule ─────────────────────────────────────────────────────────────────
ax.axhline(1.0, color=C["ink"], linewidth=0.9)

# ── column headers ────────────────────────────────────────────────────────────
for col_idx, (hdr, x, ha) in enumerate(zip(col_headers, col_x, col_ha)):
    ax.text(x, header_y, hdr,
            ha=ha, va="center",
            fontsize=9.5, fontweight="bold", color=C["ink"])

# ── mid rule (below header) ───────────────────────────────────────────────────
midrule_y = 1.0 - row_h * 1.1
ax.axhline(midrule_y, color=C["ink"], linewidth=0.5)

# ── data rows ─────────────────────────────────────────────────────────────────
for row_idx, row in enumerate(rows):
    y = midrule_y - row_h * 0.65 - row_idx * row_h
    for col_idx, (cell, x, ha) in enumerate(zip(row, col_x, col_ha)):
        ax.text(x, y, cell,
                ha=ha, va="center",
                fontsize=9.2, color=C["ink"])

# ── bottom rule ───────────────────────────────────────────────────────────────
bottom_y = midrule_y - row_h * 0.25 - n_rows * row_h
ax.axhline(bottom_y, color=C["ink"], linewidth=0.9)

fig.savefig("figures/table_dataset_stats.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/table_dataset_stats.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 2 — LABEL FREQUENCY BINS  (training split)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: label frequency bins…")

freq   = list(train_lc.values())
unseen = TOTAL_LABELS - len(train_lc)

bin_defs = [
    ("> 1000",    sum(1 for f in freq if f > 1000),       "#1E3F6F"),
    ("500 – 1000",sum(1 for f in freq if 500 < f <= 1000),"#2E5FA3"),
    ("100 – 500", sum(1 for f in freq if 100 < f <= 500), "#4A8DB5"),
    ("10 – 100",  sum(1 for f in freq if 10  < f <= 100), "#5AADA8"),
    ("1 – 10",    sum(1 for f in freq if 1   <= f <= 10), "#8DCBC7"),
    ("0  (unseen)", unseen,                                C["lgrey"]),
]
bin_labels = [b[0] for b in bin_defs]
bin_values = [b[1] for b in bin_defs]
bin_colors = [b[2] for b in bin_defs]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
bars = ax.bar(bin_labels, bin_values, color=bin_colors, width=0.55,
              linewidth=0, zorder=3)

for bar, val in zip(bars, bin_values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}",
            ha="center", va="bottom", fontsize=8.8, color=C["ink"])

ax.set_xlabel("Label frequency in training set", labelpad=7)
ax.set_ylabel("Number of EuroVoc labels", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.text(0.5, -0.02,
         f"Total EuroVoc labels: {TOTAL_LABELS:,}   ·   "
         f"Labels seen in training: {len(train_lc):,}   ·   "
         f"Unseen: {unseen:,}",
         ha="center", fontsize=8.2, color=C["grey"])

fig.tight_layout()
fig.savefig("figures/label_frequency_bins.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/label_frequency_bins.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 3 — LABELS PER DOCUMENT (grouped bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: labels per document…")

all_vals = [v for s in stats.values() for v in s["labels_per_doc_arr"]]
max_x    = min(int(np.percentile(all_vals, 99.5)), 28)
edges    = np.arange(0.5, max_x + 1.5, 1)
centers  = np.arange(1, max_x + 1)

split_names   = ["train", "validation", "test"]
split_labels  = ["Train", "Validation", "Test"]
split_colors  = ["#1E3F6F", "#4A8DB5", "#8DCBC7"]   # dark → mid → light blue

n_splits = len(split_names)
bar_w    = 0.26
offsets  = np.array([-bar_w, 0, bar_w])

fig, ax = plt.subplots(figsize=(9.0, 4.2))

for i, (split_name, label, color, offset) in enumerate(
        zip(split_names, split_labels, split_colors, offsets)):
    arr     = np.array(stats[split_name]["labels_per_doc_arr"])
    clipped = np.clip(arr, 1, max_x)
    counts, _ = np.histogram(clipped, bins=edges)
    pct     = counts / counts.sum() * 100
    n       = len(arr)
    ax.bar(centers + offset, pct, width=bar_w,
           color=color, linewidth=0, zorder=3,
           label=f"{label}  (n = {n:,})")

ax.set_xlabel("Number of labels per document", labelpad=7)
ax.set_ylabel("Documents (%)", labelpad=7)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.set_xlim(0.5, max_x + 0.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=8.8)
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/labels_per_doc.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/labels_per_doc.png")

print("\nDone. All figures saved to ./figures/")