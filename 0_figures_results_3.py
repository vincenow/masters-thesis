"""
visualize_results.py
────────────────────
Generates figures for the Results chapter. All data hardcoded from
benchmark results. Run from the repo root; figures saved to ./figures/.

Titles are intentionally omitted — captions belong in LaTeX/Overleaf.

Produces:
  1. figures/rq1_ndcg_english.png        — NDCG@5 bar chart, all models, English
  2. figures/rq2a_native_label_effect.png — EN vs native labels, grouped bars
  3. figures/summary_label_strategies.png — label representation strategies + XLM-R reference
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib import rcParams

os.makedirs("figures", exist_ok=True)

# ── palette & style ───────────────────────────────────────────────────────────
C = {
    "blue":   "#2E5FA3",
    "red":    "#B03A3A",
    "yellow": "#C9A84C",
    "grey":   "#888888",
    "lgrey":  "#DADADA",
    "ink":    "#1A1A1A",
    "bg":     "#FFFFFF",
    "navy":   "#1E3F6F",
    "steel":  "#4A8DB5",
    "teal":   "#5AADA8",
    "lteal":  "#8DCBC7",
}
FONT = "DejaVu Sans"

rcParams.update({
    "font.family":        FONT,
    "font.size":          9.5,
    "text.color":         C["ink"],
    "axes.labelcolor":    C["ink"],
    "xtick.color":        C["ink"],
    "ytick.color":        C["ink"],
    "figure.facecolor":   C["bg"],
    "axes.facecolor":     C["bg"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":     C["lgrey"],
    "axes.grid":          True,
    "grid.color":         "#EFEFEF",
    "grid.linewidth":     0.7,
    "axes.axisbelow":     True,
    "legend.frameon":     False,
    "legend.fontsize":    9,
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

# NDCG@5 and std, English, English labels, no reranking (from results CSV)
rq1_models = ["BGE-M3", "OpenAI", "E5", "Harrier", "Qwen3", "LaBSE", "GTE"]
rq1_ndcg5  = [0.1704,   0.1083,   0.0958, 0.0874,   0.0835,  0.0789,  0.0269]
rq1_std    = [0.2022,   0.1632,   0.1539, 0.1343,   0.1426,  0.1381,  0.0892]
# Note: std from NDCG_std column at k=5 in the results file

# Native-label effect (tab:rq2a)
rq2a_models = ["BGE-M3", "OpenAI", "E5", "Harrier", "Qwen3", "LaBSE", "GTE"]
rq2a = {
    #            FR_en   FR_nat  NL_en   NL_nat  DE_en   DE_nat
    "BGE-M3":  [0.1606, 0.1674, 0.1205, 0.1066, 0.1405, 0.1284],
    "OpenAI":  [0.0998, 0.1269, 0.0930, 0.1059, 0.1112, 0.1727],
    "E5":      [0.0577, 0.0662, 0.0584, 0.0530, 0.0550, 0.0641],
    "Harrier": [0.1059, 0.0949, 0.0782, 0.0871, 0.0923, 0.1744],
    "Qwen3":   [0.0638, 0.1099, 0.0453, 0.0583, 0.0588, 0.1127],
    "LaBSE":   [0.0799, 0.0816, 0.0722, 0.0677, 0.0572, 0.0181],
    "GTE":     [0.0231, 0.0324, 0.0211, 0.0282, 0.0211, 0.0268],
}

# Label representation strategies, English, top 3 models + XLM-R reference
# Strategies: Descriptor / AI-generated (EN) / Centroid / XLM-R (supervised)
strat_models = ["BGE-M3", "OpenAI", "Harrier"]
strat_data = {
    #            Desc    AI-gen  Centroid
    "BGE-M3":  [0.1704, 0.3222, 0.3949],
    "OpenAI":  [0.1083, 0.3118, 0.3930],
    "Harrier": [0.0874, 0.2323, 0.4074],
}
xlmr_english = 0.6141   # supervised reference line

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — NDCG@5 bar chart, English, all models
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 1: RQ1 NDCG@5 English…")

order    = np.argsort(rq1_ndcg5)[::-1]
models_s = [rq1_models[i] for i in order]
ndcg_s   = [rq1_ndcg5[i]  for i in order]
std_s    = [rq1_std[i]     for i in order]

bar_colors = [C["navy"] if m == "BGE-M3" else
              C["lgrey"] if m == "GTE" else
              C["steel"] for m in models_s]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(models_s))
bars = ax.bar(x, ndcg_s, color=bar_colors, width=0.55,
              linewidth=0, zorder=3)

for bar, val in zip(bars, ndcg_s):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8.2, color=C["ink"])

ax.set_xticks(x)
ax.set_xticklabels(models_s, fontsize=9)
ax.set_ylabel("NDCG@5", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_ylim(0, max(ndcg_s) + 0.03)
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/rq1_ndcg_english.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/rq1_ndcg_english.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Native-label effect, grouped bars per model
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: native-label effect…")

group_w    = 0.12
gaps       = [0, group_w, 3*group_w, 4*group_w, 6*group_w, 7*group_w]
group_span = 8 * group_w
model_positions = np.arange(len(rq2a_models)) * (group_span + 0.25)
bar_cols   = [C["steel"], C["navy"], C["steel"], C["teal"],
              C["steel"], C["yellow"]]

fig, ax = plt.subplots(figsize=(10.0, 4.2))

for m_idx, model in enumerate(rq2a_models):
    vals = rq2a[model]
    base = model_positions[m_idx]
    for gap, val, col in zip(gaps, vals, bar_cols):
        ax.bar(base + gap, val, width=group_w, color=col, linewidth=0, zorder=3)

centres = model_positions + group_span / 2 - group_w / 2
ax.set_xticks(centres)
ax.set_xticklabels(rq2a_models, fontsize=9)
ax.set_ylabel("NDCG@5", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

legend_elements = [
    Patch(facecolor=C["steel"],  label="EN labels"),
    Patch(facecolor=C["navy"],   label="FR native"),
    Patch(facecolor=C["teal"],   label="NL native"),
    Patch(facecolor=C["yellow"], label="DE native"),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc="upper right", ncol=2)

fig.tight_layout()
fig.savefig("figures/rq2a_native_label_effect.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/rq2a_native_label_effect.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Label representation strategies + XLM-R reference line
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: label representation strategies…")

strat_labels = ["Descriptor\n(baseline)", "AI-generated\ndescriptor", "Centroid\n(partial supervision)"]
strat_colors = [C["steel"], C["navy"], C["teal"]]

n_models  = len(strat_models)
n_strats  = len(strat_labels)
bar_w     = 0.22
group_gap = 0.12
positions = np.arange(n_models) * (n_strats * bar_w + group_gap)

fig, ax = plt.subplots(figsize=(7.2, 4.4))

for s_idx, (label, color) in enumerate(zip(strat_labels, strat_colors)):
    vals = [strat_data[m][s_idx] for m in strat_models]
    xpos = positions + s_idx * bar_w
    bars = ax.bar(xpos, vals, width=bar_w, color=color,
                  linewidth=0, zorder=3, label=label)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.8, color=C["ink"])

# XLM-R supervised reference line
x_min = positions[0] - bar_w
x_max = positions[-1] + n_strats * bar_w
ax.hlines(xlmr_english, x_min, x_max,
          colors=C["red"], linewidths=1.4, linestyles="--", zorder=4)
ax.text(x_max + 0.02, xlmr_english, f"XLM-R (supervised)\n{xlmr_english:.4f}",
        va="center", ha="left", fontsize=8.0, color=C["red"])

centres = positions + bar_w * (n_strats - 1) / 2
ax.set_xticks(centres)
ax.set_xticklabels(strat_models, fontsize=9.5)
ax.set_ylabel("NDCG@5", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_ylim(0, 0.72)
ax.set_xlim(x_min - 0.05, x_max + 1.1)
ax.legend(fontsize=8.8, loc="upper center", ncol=3, bbox_to_anchor=(0.38, 1.0))
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/summary_label_strategies.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/summary_label_strategies.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — AI-generated descriptor gain, all 6 models, English
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: RQ2b generated descriptor gain…")

gen_models  = ["BGE-M3", "OpenAI", "Qwen3", "Harrier", "E5", "LaBSE"]
gen_desc    = [0.1704,    0.1083,   0.0835,  0.0874,    0.0958, 0.0789]
gen_ai      = [0.3222,    0.3118,   0.3104,  0.2323,    0.1311, 0.0811]

n = len(gen_models)
bar_w = 0.32
x = np.arange(n)

fig, ax = plt.subplots(figsize=(8.0, 4.0))

bars_desc = ax.bar(x - bar_w/2, gen_desc, width=bar_w, color=C["steel"],
                   linewidth=0, zorder=3, label="EuroVoc descriptor")
bars_ai   = ax.bar(x + bar_w/2, gen_ai,   width=bar_w, color=C["navy"],
                   linewidth=0, zorder=3, label="AI-generated descriptor")

# value labels
for bar, val in zip(bars_desc, gen_desc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.0, color=C["ink"])
for bar, val in zip(bars_ai, gen_ai):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.0, color=C["ink"])

ax.set_xticks(x)
ax.set_xticklabels(gen_models, fontsize=9.5)
ax.set_ylabel("NDCG@5", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_ylim(0, 0.40)
ax.legend(fontsize=8.8, loc="upper right")
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/rq2b_generated_gain.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/rq2b_generated_gain.png")

print("\nDone. All figures saved to ./figures/")