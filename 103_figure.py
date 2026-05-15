"""
XMTC Retrieval Results – Visualization Dashboard
=================================================
Three panels, each showing a different comparison:
  Panel A: Model comparison (English docs, English labels, base retrieval)
  Panel B: Reranking effect (base vs reranked, English docs, English labels)
  Panel C: Label language effect (English labels vs native labels, base retrieval)

Each panel contains 3 sub-plots: Precision@k, Recall@k, NDCG@k
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def metrics(data, metric, k_values):
    return [data["metrics"][metric][str(k)]["mean"] for k in k_values]

# ── data ─────────────────────────────────────────────────────────────────────

K = [5, 10, 20, 50, 100]

FILES = {
    # base English
    "E5":      "results_e5_small_english.json",
    "LaBSE":   "results_labse_english.json",
    "OpenAI":  "results_openai_english.json",
    "GTE":     "results_gte_en_en_labels.json",
    "BGE-M3":  "results_bge_m3_en_en_labels.json",
    "BM25":    "results_bm25_en_en_labels.json",

    # reranked English
    "E5 (reranked)":     "results_e5_reranked_en_en_labels.json",
    "LaBSE (reranked)":  "results_labse_reranked_en_en_labels.json",
    "OpenAI (reranked)": "results_openai_reranked_en_en_labels.json",
    "GTE (reranked)":    "results_gte_reranked_en_en_labels.json",
    "BGE-M3 (reranked)": "results_bge_m3_reranked_en_en_labels.json",
    "BM25 (reranked)":   "results_bm25_reranked_en_en_labels.json",

    # French – English labels vs native
    "E5 French (EN labels)":     "results_e5_french_enlabels.json",
    "E5 French (native)":        "results_e5_french_nativelabels.json",
    "LaBSE French (EN labels)":  "results_labse_french_enlabels.json",
    "LaBSE French (native)":     "results_labse_french_nativelabels.json",
    "OpenAI French (EN labels)": "results_openai_french_enlabels.json",
    "OpenAI French (native)":    "results_openai_french_nativelabels.json",
    "GTE French (EN labels)":    "results_gte_fr_en_labels.json",
    "GTE French (native)":       "results_gte_fr_native_labels.json",
    "BGE-M3 French (EN labels)": "results_bge_m3_fr_en_labels.json",
    "BGE-M3 French (native)":    "results_bge_m3_fr_native_labels.json",
}

data = {}
for name, path in FILES.items():
    if os.path.exists(path):
        data[name] = load(path)
    else:
        print(f"[missing] {path}")

def get(name, metric):
    if name not in data:
        return None
    return metrics(data[name], metric, K)

# ── style ─────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "E5":     "#2563EB",   # blue
    "LaBSE":  "#16A34A",   # green
    "OpenAI": "#DC2626",   # red
    "GTE":    "#9333EA",   # purple
    "BGE-M3": "#EA580C",   # orange
    "BM25":   "#6B7280",   # grey
}

METRICS = [
    ("precision", "Precision@k"),
    ("recall",    "Recall@k"),
    ("ndcg",      "NDCG@k"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
})

# ── figure layout ─────────────────────────────────────────────────────────────
# 3 rows (panels A / B / C), 3 columns (P / R / NDCG)
# + a narrow legend column on the right

fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor("#F8FAFC")

outer = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.06, right=0.99,
    top=0.94, bottom=0.05,
    hspace=0.45, wspace=0.08,
    width_ratios=[1, 0.18],
)

panel_labels = [
    "A  –  Model comparison  (English documents, English labels, base retrieval)",
    "B  –  Reranking effect  (English documents, English labels)",
    "C  –  Label language effect  (French documents, base retrieval)",
]

axes = []  # axes[panel][metric_idx]
for row in range(3):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[row, 0], wspace=0.32
    )
    row_axes = [fig.add_subplot(inner[0, col]) for col in range(3)]
    axes.append(row_axes)
    # panel label
    fig.text(
        0.01, 1 - row / 3 - 0.01,
        panel_labels[row],
        transform=fig.transFigure,
        fontsize=8.5, fontweight="bold", color="#1E293B",
        va="top",
    )

legend_axes = [fig.add_subplot(outer[row, 1]) for row in range(3)]
for ax in legend_axes:
    ax.axis("off")

# ── helpers for plotting ──────────────────────────────────────────────────────

def plot_line(ax, name, metric, color, linestyle="-", label=None):
    y = get(name, metric)
    if y is None:
        return
    ax.plot(K, y, color=color, linestyle=linestyle,
            marker="o", label=label or name)

def style_ax(ax, title, ylabel=True):
    ax.set_title(title, pad=4)
    ax.set_xlabel("k")
    ax.set_xticks(K)
    ax.set_xticklabels(K)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(0.7)
    if ylabel:
        ax.set_ylabel("Score")

# ── PANEL A: model comparison ─────────────────────────────────────────────────

models_A = ["E5", "LaBSE", "OpenAI", "GTE", "BGE-M3", "BM25"]
for col, (metric_key, metric_label) in enumerate(METRICS):
    ax = axes[0][col]
    for m in models_A:
        plot_line(ax, m, metric_key, MODEL_COLORS[m], label=m)
    style_ax(ax, metric_label, ylabel=(col == 0))

legend_handles_A = [
    Line2D([0], [0], color=MODEL_COLORS[m], marker="o",
           markersize=4, linewidth=1.6, label=m)
    for m in models_A
]
legend_axes[0].legend(
    handles=legend_handles_A, loc="center left",
    title="Model", title_fontsize=8, frameon=True,
    facecolor="#F1F5F9", edgecolor="#CBD5E1",
)

# ── PANEL B: reranking effect ─────────────────────────────────────────────────

for col, (metric_key, metric_label) in enumerate(METRICS):
    ax = axes[1][col]
    for m in models_A:
        c = MODEL_COLORS[m]
        plot_line(ax, m,                metric_key, c, linestyle="-",  label=f"{m} base")
        plot_line(ax, f"{m} (reranked)", metric_key, c, linestyle="--", label=f"{m} reranked")
    style_ax(ax, metric_label, ylabel=(col == 0))

legend_handles_B = []
for m in models_A:
    legend_handles_B.append(
        Line2D([0], [0], color=MODEL_COLORS[m], marker="o",
               markersize=4, linewidth=1.6, label=m)
    )
legend_handles_B += [
    Line2D([0], [0], color="#374151", linestyle="-",  linewidth=1.6, label="Base"),
    Line2D([0], [0], color="#374151", linestyle="--", linewidth=1.6, label="Reranked"),
]
legend_axes[1].legend(
    handles=legend_handles_B, loc="center left",
    title="Model / Stage", title_fontsize=8, frameon=True,
    facecolor="#F1F5F9", edgecolor="#CBD5E1",
)

# ── PANEL C: label language effect (French docs) ──────────────────────────────

models_C = ["E5", "LaBSE", "OpenAI", "GTE", "BGE-M3"]
for col, (metric_key, metric_label) in enumerate(METRICS):
    ax = axes[2][col]
    for m in models_C:
        c = MODEL_COLORS[m]
        plot_line(ax, f"{m} French (EN labels)", metric_key, c,
                  linestyle="-",  label=f"{m} EN labels")
        plot_line(ax, f"{m} French (native)",    metric_key, c,
                  linestyle="--", label=f"{m} native")
    style_ax(ax, metric_label, ylabel=(col == 0))

legend_handles_C = []
for m in models_C:
    legend_handles_C.append(
        Line2D([0], [0], color=MODEL_COLORS[m], marker="o",
               markersize=4, linewidth=1.6, label=m)
    )
legend_handles_C += [
    Line2D([0], [0], color="#374151", linestyle="-",  linewidth=1.6, label="EN labels"),
    Line2D([0], [0], color="#374151", linestyle="--", linewidth=1.6, label="Native labels"),
]
legend_axes[2].legend(
    handles=legend_handles_C, loc="center left",
    title="Model / Label lang.", title_fontsize=8, frameon=True,
    facecolor="#F1F5F9", edgecolor="#CBD5E1",
)

# ── title & save ──────────────────────────────────────────────────────────────

fig.suptitle(
    "XMTC Retrieval Results  –  MultiEURLEX / EuroVoc",
    fontsize=12, fontweight="bold", color="#0F172A", y=0.975,
)

# Save
out_path = "/Users/vincent/masters-thesis/xmtc_results_dashboard.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")

plt.show()