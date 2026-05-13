"""
Thesis presentation plots — zero-shot XMTC on MultiEURLEX
Reproduces the 4 tabs from the interactive dashboard:
  1. Model ranking (NDCG@5, English)
  2. Reranking effect (retrieval vs reranked, English)
  3. Native label effect per language (RQ2)
  4. NDCG@k curves (English, retrieval only)

Run: python thesis_presentation_plots.py
Requires: matplotlib, numpy
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Data loading  (mirrors your existing load_results / DataFrame logic)
# ---------------------------------------------------------------------------

FILES = {
    "E5 English":                   "results_e5_small_english.json",
    "E5 French":                    "results_e5_french_enlabels.json",
    "E5 French (native labels)":    "results_e5_french_nativelabels.json",
    "E5 Dutch":                     "results_e5_dutch_enlabels.json",
    "E5 Dutch (native labels)":     "results_e5_dutch_nativelabels.json",
    "E5 German":                    "results_e5_german_enlabels.json",
    "E5 German (native labels)":    "results_e5_german_nativelabels.json",
    "E5 English (reranked)":        "results_e5_reranked_en_en_labels.json",
    "E5 French native (reranked)":  "results_e5_reranked_fr_native_labels.json",
    "E5 Dutch native (reranked)":   "results_e5_reranked_nl_native_labels.json",
    "E5 German native (reranked)":  "results_e5_reranked_de_native_labels.json",

    "LaBSE English":                "results_labse_english.json",
    "LaBSE French":                 "results_labse_french_enlabels.json",
    "LaBSE French (native labels)": "results_labse_french_nativelabels.json",
    "LaBSE Dutch":                  "results_labse_dutch_enlabels.json",
    "LaBSE Dutch (native labels)":  "results_labse_dutch_nativelabels.json",
    "LaBSE German":                 "results_labse_german_enlabels.json",
    "LaBSE German (native labels)": "results_labse_german_nativelabels.json",
    "LaBSE English (reranked)":     "results_labse_reranked_en_en_labels.json",

    "OpenAI English":               "results_openai_english.json",
    "OpenAI French":                "results_openai_french_enlabels.json",
    "OpenAI French (native labels)":"results_openai_french_nativelabels.json",
    "OpenAI Dutch":                 "results_openai_dutch_enlabels.json",
    "OpenAI Dutch (native labels)": "results_openai_dutch_nativelabels.json",
    "OpenAI German":                "results_openai_german_enlabels.json",
    "OpenAI German (native labels)":"results_openai_german_nativelabels.json",
    "OpenAI English (reranked)":    "results_openai_reranked_en_en_labels.json",
    "OpenAI French native (reranked)": "results_openai_reranked_fr_native_labels.json",
    "OpenAI Dutch native (reranked)":  "results_openai_reranked_nl_native_labels.json",
    "OpenAI German native (reranked)": "results_openai_reranked_de_native_labels.json",

    "GTE English":                  "results_gte_en_en_labels.json",
    "GTE French":                   "results_gte_fr_en_labels.json",
    "GTE French (native labels)":   "results_gte_fr_native_labels.json",
    "GTE Dutch":                    "results_gte_nl_en_labels.json",
    "GTE Dutch (native labels)":    "results_gte_nl_native_labels.json",
    "GTE German":                   "results_gte_de_en_labels.json",
    "GTE German (native labels)":   "results_gte_de_native_labels.json",
    "GTE English (reranked)":       "results_gte_reranked_en_en_labels.json",

    "BGE-M3 English":               "results_bge_m3_en_en_labels.json",
    "BGE-M3 French":                "results_bge_m3_fr_en_labels.json",
    "BGE-M3 French (native labels)":"results_bge_m3_fr_native_labels.json",
    "BGE-M3 Dutch":                 "results_bge_m3_nl_en_labels.json",
    "BGE-M3 Dutch (native labels)": "results_bge_m3_nl_native_labels.json",
    "BGE-M3 German":                "results_bge_m3_de_en_labels.json",
    "BGE-M3 German (native labels)":"results_bge_m3_de_native_labels.json",
    "BGE-M3 English (reranked)":    "results_bge_m3_reranked_en_en_labels.json",

    "BM25 English":                 "results_bm25_en_en_labels.json",
    "BM25 English (reranked)":      "results_bm25_reranked_en_en_labels.json",
}

K_VALUES = [5, 10, 20, 50, 100]


def load_ndcg(name, k=5):
    """Return NDCG@k mean for a named model config, or None if file missing."""
    path = FILES.get(name)
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]["ndcg"][str(k)]["mean"]


def load_ndcg_curve(name):
    """Return list of NDCG means for all K_VALUES."""
    return [load_ndcg(name, k) for k in K_VALUES]


# ---------------------------------------------------------------------------
# Colour palette (matches interactive dashboard)
# ---------------------------------------------------------------------------

C = {
    "bge":     "#3B6D11",
    "openai":  "#185FA5",
    "e5":      "#3C3489",
    "labse":   "#993C1D",
    "gte":     "#888780",
    "bm25":    "#444441",
    "reranked":"#E24B4A",
    "native":  "#3B6D11",
    "en":      "#185FA5",
}

MODEL_COLORS = {
    "BGE-M3": C["bge"],
    "OpenAI":  C["openai"],
    "E5":      C["e5"],
    "LaBSE":   C["labse"],
    "GTE":     C["gte"],
    "BM25":    C["bm25"],
}

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 130,
})

LABEL_FS = 9
TICK_FS  = 8
TITLE_FS = 10


def annotate_bars(ax, bars, fmt="{:.3f}", offset=0.002, fontsize=8):
    for bar in bars:
        v = bar.get_width() if hasattr(bar, "get_width") and bar.get_width() != bar.get_height() else bar.get_height()
        x = bar.get_x() + bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if bar.get_width() > bar.get_height():  # horizontal bar
            ax.text(x + offset, y, fmt.format(v), va="center", ha="left", fontsize=fontsize)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    fmt.format(v), ha="center", va="bottom", fontsize=fontsize)


# ---------------------------------------------------------------------------
# Plot 1 — Model ranking
# ---------------------------------------------------------------------------

def plot_ranking(ax):
    models = ["BGE-M3", "OpenAI", "E5", "LaBSE", "GTE", "BM25"]
    keys   = ["BGE-M3 English", "OpenAI English", "E5 English",
               "LaBSE English", "GTE English", "BM25 English"]
    values = [load_ndcg(k) or 0 for k in keys]
    colors = [MODEL_COLORS[m] for m in models]

    y = np.arange(len(models))
    bars = ax.barh(y, values, color=colors, height=0.55, zorder=3)

    for bar, v in zip(bars, values):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=TICK_FS)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=LABEL_FS)
    ax.set_xlim(0, 0.21)
    ax.set_xlabel("NDCG@5", fontsize=LABEL_FS)
    ax.set_title("Model ranking — English, retrieval only", fontsize=TITLE_FS, pad=8)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Plot 2 — Reranking effect
# ---------------------------------------------------------------------------

def plot_reranking(ax):
    models    = ["BGE-M3", "OpenAI", "E5", "LaBSE", "GTE", "BM25"]
    retrieval = [load_ndcg(k) or 0 for k in
                 ["BGE-M3 English", "OpenAI English", "E5 English",
                  "LaBSE English",  "GTE English",    "BM25 English"]]
    reranked  = [load_ndcg(k) or 0 for k in
                 ["BGE-M3 English (reranked)", "OpenAI English (reranked)",
                  "E5 English (reranked)",     "LaBSE English (reranked)",
                  "GTE English (reranked)",    "BM25 English (reranked)"]]

    x   = np.arange(len(models))
    w   = 0.35
    b1  = ax.bar(x - w/2, retrieval, width=w, color=C["en"],      label="Retrieval", zorder=3)
    b2  = ax.bar(x + w/2, reranked,  width=w, color=C["reranked"], label="Reranked",  zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=LABEL_FS)
    ax.set_ylabel("NDCG@5", fontsize=LABEL_FS)
    ax.set_ylim(0, 0.22)
    ax.set_title("Reranking effect — English, k=5", fontsize=TITLE_FS, pad=8)
    ax.legend(fontsize=TICK_FS, frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Plot 3 — Native label effect (RQ2), one sub-panel per language
# ---------------------------------------------------------------------------

NATIVE_DATA = {
    "French": {
        "models": ["E5", "BGE-M3", "OpenAI", "LaBSE", "GTE"],
        "en":     ["E5 French", "BGE-M3 French", "OpenAI French",
                   "LaBSE French", "GTE French"],
        "native": ["E5 French (native labels)", "BGE-M3 French (native labels)",
                   "OpenAI French (native labels)", "LaBSE French (native labels)",
                   "GTE French (native labels)"],
    },
    "German": {
        "models": ["E5", "BGE-M3", "OpenAI", "LaBSE", "GTE"],
        "en":     ["E5 German", "BGE-M3 German", "OpenAI German",
                   "LaBSE German", "GTE German"],
        "native": ["E5 German (native labels)", "BGE-M3 German (native labels)",
                   "OpenAI German (native labels)", "LaBSE German (native labels)",
                   "GTE German (native labels)"],
    },
    "Dutch": {
        "models": ["E5", "BGE-M3", "OpenAI", "LaBSE", "GTE"],
        "en":     ["E5 Dutch", "BGE-M3 Dutch", "OpenAI Dutch",
                   "LaBSE Dutch", "GTE Dutch"],
        "native": ["E5 Dutch (native labels)", "BGE-M3 Dutch (native labels)",
                   "OpenAI Dutch (native labels)", "LaBSE Dutch (native labels)",
                   "GTE Dutch (native labels)"],
    },
}


def plot_native(axes):
    langs = ["French", "German", "Dutch"]
    for ax, lang in zip(axes, langs):
        d  = NATIVE_DATA[lang]
        en = [load_ndcg(k) or 0 for k in d["en"]]
        nt = [load_ndcg(k) or 0 for k in d["native"]]
        x  = np.arange(len(d["models"]))
        w  = 0.35

        ax.bar(x - w/2, en, width=w, color=C["en"],     label="English labels", zorder=3)
        ax.bar(x + w/2, nt, width=w, color=C["native"], label="Native labels",  zorder=3)

        # highlight LaBSE if it collapses
        for i, (e, n) in enumerate(zip(en, nt)):
            if d["models"][i] == "LaBSE" and n < e * 0.5:
                ax.bar(x[i] + w/2, n, width=w, color="#E24B4A", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(d["models"], fontsize=TICK_FS, rotation=90, ha="center")
        ax.set_title(lang, fontsize=TITLE_FS, pad=6)
        ax.set_ylim(0, 0.20)
        ax.set_ylabel("NDCG@5" if lang == "French" else "", fontsize=LABEL_FS)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    legend_handles = [
        mpatches.Patch(color=C["en"],     label="English labels"),
        mpatches.Patch(color=C["native"], label="Native labels"),
        mpatches.Patch(color="#E24B4A",   label="LaBSE collapse"),
    ]
    axes[-1].legend(handles=legend_handles, fontsize=TICK_FS, frameon=False,
                    loc="upper right")


# ---------------------------------------------------------------------------
# Plot 4 — NDCG@k curves
# ---------------------------------------------------------------------------

CURVE_CONFIGS = [
    ("BGE-M3 English", "BGE-M3", C["bge"],    "-",   "o"),
    ("OpenAI English", "OpenAI",  C["openai"], "--",  "s"),
    ("E5 English",     "E5",      C["e5"],     "-.",  "^"),
    ("LaBSE English",  "LaBSE",   C["labse"],  ":",   "D"),
    ("GTE English",    "GTE",     C["gte"],    "--",  "x"),
    ("BM25 English",   "BM25",    C["bm25"],   ":",   "."),
]


def plot_curves(ax):
    for key, label, color, ls, marker in CURVE_CONFIGS:
        curve = load_ndcg_curve(key)
        vals  = [v if v is not None else 0 for v in curve]
        ax.plot(K_VALUES, vals, color=color, linestyle=ls, marker=marker,
                markersize=5, linewidth=1.8, label=label, zorder=3)

    ax.set_xlabel("k", fontsize=LABEL_FS)
    ax.set_ylabel("NDCG@k", fontsize=LABEL_FS)
    ax.set_title("NDCG@k curves — English, retrieval only", fontsize=TITLE_FS, pad=8)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=TICK_FS, frameon=False, ncol=2)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Compose figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 11))
fig.suptitle("Zero-shot XMTC on MultiEURLEX — preliminary results",
             fontsize=13, fontweight="bold", y=0.98)

# Layout: 2×2 grid; bottom row has 3 sub-panels for native labels
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

ax_rank    = fig.add_subplot(gs[0, 0])
ax_rerank  = fig.add_subplot(gs[0, 1])
ax_curves  = fig.add_subplot(gs[1, 1])

# Native labels: 3 sub-panels sharing the bottom-left cell
gs_native  = gs[1, 0].subgridspec(1, 3, wspace=0.45)
ax_native  = [fig.add_subplot(gs_native[0, i]) for i in range(3)]

plot_ranking(ax_rank)
plot_reranking(ax_rerank)
plot_native(ax_native)
plot_curves(ax_curves)

# Panel labels
for ax, label in zip(
    [ax_rank, ax_rerank, ax_native[0], ax_curves],
    ["A", "B", "C", "D"]
):
    ax.text(-0.12, 1.07, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

plt.savefig("xmtc_results.png", dpi=150, bbox_inches="tight")
print("Saved: xmtc_results.png")
plt.show()