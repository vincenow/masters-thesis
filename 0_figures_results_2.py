"""
visualize_results2.py
─────────────────────
Generates four additional figures for the Results chapter.
All data hardcoded from benchmark results. Run from repo root;
figures saved to ./figures/.

Titles intentionally omitted — captions belong in LaTeX/Overleaf.

Produces:
  4. figures/ndcg_at_k_curves.png        — NDCG@k line plot, all models, k=5..100
  5. figures/reranking_delta.png         — reranking Δ NDCG@5 bar chart
  6. figures/bm25_comparison.png         — BM25 doc-query vs label-query by language
  7. figures/heatmap_rq1_rq2a.png        — NDCG@5 heatmap, models × (lang × label cond)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
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

K_VALUES = [5, 10, 20, 50, 100]

# NDCG@k, English test set, English labels, no reranking
ndcg_at_k = {
    "BGE-M3":  [0.1704, 0.1742, 0.1977, 0.2268, 0.2540],
    "OpenAI":  [0.1083, 0.1134, 0.1329, 0.1632, 0.1852],
    "E5":      [0.0958, 0.1088, 0.1375, 0.1782, 0.2098],
    "Harrier": [0.0874, 0.0933, 0.1082, 0.1250, 0.1396],
    "Qwen3":   [0.0835, 0.0900, 0.1072, 0.1341, 0.1597],
    "LaBSE":   [0.0789, 0.0750, 0.0849, 0.1022, 0.1194],
    "GTE":     [0.0269, 0.0308, 0.0389, 0.0523, 0.0654],
}

# Reranking: NDCG@5 before and after, English labels, English docs
reranking_models = ["BGE-M3", "OpenAI", "E5", "Qwen3", "Harrier", "LaBSE", "GTE", "BM25"]
reranking_before = [0.1704,   0.1083,   0.0958, 0.0835, 0.0874,   0.0789,  0.0269, 0.0849]
reranking_after  = [0.0914,   0.0807,   0.1135, 0.1156, 0.0850,   0.0366,  0.0788, 0.0807]

# BM25 NDCG@5 by language and condition
bm25_languages = ["English", "French", "Dutch", "German"]
bm25_data = {
    # [std_en, lq_en, lq_native]
    "English": [0.0849, 0.2382, None],
    "French":  [0.0014, 0.0793, 0.2410],
    "Dutch":   [0.0125, 0.0767, 0.1833],
    "German":  [0.0464, 0.0604, 0.1941],
}

# Heatmap data
heatmap_models = ["BGE-M3", "OpenAI", "Harrier", "Qwen3", "E5", "LaBSE", "GTE"]
heatmap_cols   = ["EN\n(EN)", "FR\n(EN)", "FR\n(nat)", "NL\n(EN)", "NL\n(nat)", "DE\n(EN)", "DE\n(nat)"]
heatmap_data   = np.array([
    [0.1704, 0.1606, 0.1674, 0.1205, 0.1066, 0.1405, 0.1284],
    [0.1083, 0.0998, 0.1269, 0.0930, 0.1059, 0.1112, 0.1727],
    [0.0874, 0.1059, 0.0949, 0.0782, 0.0871, 0.0923, 0.1744],
    [0.0835, 0.0638, 0.1099, 0.0453, 0.0583, 0.0588, 0.1127],
    [0.0958, 0.0577, 0.0662, 0.0584, 0.0530, 0.0550, 0.0641],
    [0.0789, 0.0799, 0.0816, 0.0722, 0.0677, 0.0572, 0.0181],
    [0.0269, 0.0231, 0.0324, 0.0211, 0.0282, 0.0211, 0.0268],
])

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — NDCG@k curves, all models
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: NDCG@k curves…")

# all solid lines, distinct colours and markers only — no dashes
model_styles = {
    "BGE-M3":  dict(color=C["navy"],   lw=2.2, marker="o"),
    "OpenAI":  dict(color=C["blue"],   lw=1.6, marker="s"),
    "E5":      dict(color=C["teal"],   lw=1.6, marker="^"),
    "Harrier": dict(color=C["yellow"], lw=1.6, marker="D"),
    "Qwen3":   dict(color=C["red"],    lw=1.6, marker="v"),
    "LaBSE":   dict(color=C["steel"],  lw=1.6, marker="P"),
    "GTE":     dict(color=C["lgrey"],  lw=1.4, marker="x"),
}

fig, ax = plt.subplots(figsize=(7.2, 4.4))

for model, vals in ndcg_at_k.items():
    sty = model_styles[model]
    ax.plot(K_VALUES, vals,
            color=sty["color"], linewidth=sty["lw"],
            linestyle="-", marker=sty["marker"],
            markersize=4, label=model, zorder=3)

ax.set_xlabel("k", labelpad=7)
ax.set_ylabel("NDCG@k", labelpad=7)
ax.set_xticks(K_VALUES)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(fontsize=8.5, loc="upper left", ncol=2)
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/ndcg_at_k_curves.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/ndcg_at_k_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Reranking Δ NDCG@5
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 5: reranking delta…")

deltas   = [a - b for a, b in zip(reranking_after, reranking_before)]
order    = np.argsort(deltas)[::-1]
models_s = [reranking_models[i] for i in order]
deltas_s = [deltas[i] for i in order]
colors_s = [C["teal"] if d >= 0 else C["red"] for d in deltas_s]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(models_s))
ax.bar(x, deltas_s, color=colors_s, width=0.55, linewidth=0, zorder=3)

for i, val in enumerate(deltas_s):
    va   = "bottom" if val >= 0 else "top"
    ypos = val + 0.002 if val >= 0 else val - 0.002
    ax.text(i, ypos, f"{val:+.3f}",
            ha="center", va=va, fontsize=8.2, color=C["ink"])

ax.axhline(0, color=C["ink"], linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models_s, fontsize=9)
ax.set_ylabel("Δ NDCG@5  (reranked − base)", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)
ax.legend(handles=[Patch(facecolor=C["teal"], label="Improvement"),
                   Patch(facecolor=C["red"],  label="Degradation")],
          fontsize=8.5, loc="lower left")

fig.tight_layout()
fig.savefig("figures/reranking_delta.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/reranking_delta.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — BM25 directional comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 6: BM25 comparison…")

std_en    = [bm25_data[l][0] for l in bm25_languages]
lq_en     = [bm25_data[l][1] for l in bm25_languages]
lq_native = [bm25_data[l][2] for l in bm25_languages]

bar_w = 0.22
x     = np.arange(len(bm25_languages))

fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.bar(x - bar_w, std_en, width=bar_w, color=C["steel"], linewidth=0,
       zorder=3, label="Standard BM25 (EN labels)")
ax.bar(x,         lq_en,  width=bar_w, color=C["navy"],  linewidth=0,
       zorder=3, label="BM25-LQ (EN labels)")

# native bars: draw individually, skip English (None)
for i, val in enumerate(lq_native):
    if val is not None:
        ax.bar(x[i] + bar_w, val, width=bar_w, color=C["lteal"],
               linewidth=0, zorder=3)

# add legend entry for native manually
ax.bar([], [], color=C["lteal"], linewidth=0, label="BM25-LQ (native labels)")

# value labels
for i in range(len(bm25_languages)):
    ax.text(x[i] - bar_w, std_en[i] + 0.004, f"{std_en[i]:.3f}",
            ha="center", va="bottom", fontsize=7.8, color=C["ink"])
    ax.text(x[i],         lq_en[i]  + 0.004, f"{lq_en[i]:.3f}",
            ha="center", va="bottom", fontsize=7.8, color=C["ink"])
    if lq_native[i] is not None:
        ax.text(x[i] + bar_w, lq_native[i] + 0.004, f"{lq_native[i]:.3f}",
                ha="center", va="bottom", fontsize=7.8, color=C["ink"])

ax.set_xticks(x)
ax.set_xticklabels(bm25_languages, fontsize=9.5)
ax.set_ylabel("NDCG@5", labelpad=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(fontsize=8.5, loc="upper right")
ax.spines["bottom"].set_linewidth(0.6)
ax.spines["left"].set_linewidth(0.6)

fig.tight_layout()
fig.savefig("figures/bm25_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/bm25_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 7: heatmap…")

fig, ax = plt.subplots(figsize=(8.0, 3.8))

cmap = mcolors.LinearSegmentedColormap.from_list(
    "thesis_blue", ["#FFFFFF", C["steel"], C["navy"]], N=256)
im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.20)

for r in range(len(heatmap_models)):
    for c in range(len(heatmap_cols)):
        val = heatmap_data[r, c]
        txt_color = "white" if val > 0.12 else C["ink"]
        ax.text(c, r, f"{val:.3f}",
                ha="center", va="center", fontsize=7.8, color=txt_color)

ax.set_xticks(range(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, fontsize=8.5)
ax.set_yticks(range(len(heatmap_models)))
ax.set_yticklabels(heatmap_models, fontsize=9)

for sep in [1.5, 3.5, 5.5]:
    ax.axvline(sep, color=C["lgrey"], linewidth=1.2)

for label, cx in [("English", 0), ("French", 1.5), ("Dutch", 3.5), ("German", 5.5)]:
    ax.text(cx, -0.85, label, ha="center", va="top",
            fontsize=8.5, color=C["grey"],
            transform=ax.get_xaxis_transform())

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("NDCG@5", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

ax.spines[:].set_visible(False)
ax.tick_params(bottom=False, left=False)

fig.tight_layout()
fig.savefig("figures/heatmap_rq1_rq2a.png", dpi=180, bbox_inches="tight")
plt.close()
print("  → figures/heatmap_rq1_rq2a.png")

print("\nDone. All figures saved to ./figures/")