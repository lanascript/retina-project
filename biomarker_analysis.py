# biomarker_analysis.py
#
# Run AFTER bangladesh_biomarkers.py has produced bangladesh_biomarkers.csv
# Produces:
#   - biomarker_boxplots.png   (3-panel figure, publication-ready)
#   - biomarker_stats.txt      (Mann-Whitney U test results)
#   - prints a summary table for your report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

CSV_PATH   = "bangladesh_biomarkers.csv"
PLOT_PATH  = "biomarker_boxplots.png"
STATS_PATH = "biomarker_stats.txt"

# =====================================================================
# LOAD
# =====================================================================
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Classes: {df['class'].value_counts().to_dict()}\n")

normal = df[df["class"] == "Normal"]
papil  = df[df["class"] == "Papilledema"]

biomarkers = {
    "vessel_density":    "Vessel Density",
    "tortuosity":        "Mean Tortuosity",
    "fractal_dimension": "Fractal Dimension",
}

# =====================================================================
# STATS
# =====================================================================
stats_lines = ["Biomarker Statistical Analysis — Normal vs Papilledema\n",
               "="*60 + "\n"]

results_table = []

for col, label in biomarkers.items():
    n_vals = normal[col].dropna().values
    p_vals = papil[col].dropna().values

    if len(n_vals) < 2 or len(p_vals) < 2:
        print(f"WARNING: not enough values for {col}, skipping stats")
        continue

    stat, pvalue = mannwhitneyu(n_vals, p_vals, alternative="two-sided")

    sig = ""
    if pvalue < 0.001:   sig = "***"
    elif pvalue < 0.01:  sig = "**"
    elif pvalue < 0.05:  sig = "*"
    else:                sig = "ns"

    line = (
        f"{label}:\n"
        f"  Normal      : mean={n_vals.mean():.4f}  std={n_vals.std():.4f}  n={len(n_vals)}\n"
        f"  Papilledema : mean={p_vals.mean():.4f}  std={p_vals.std():.4f}  n={len(p_vals)}\n"
        f"  Mann-Whitney U={stat:.1f}  p={pvalue:.4f}  {sig}\n"
    )
    stats_lines.append(line)
    print(line)

    results_table.append({
        "Biomarker":         label,
        "Normal mean±std":   f"{n_vals.mean():.4f} ± {n_vals.std():.4f}",
        "Papill mean±std":   f"{p_vals.mean():.4f} ± {p_vals.std():.4f}",
        "p-value":           f"{pvalue:.4f}",
        "Significance":      sig,
    })

with open(STATS_PATH, "w") as f:
    f.writelines(stats_lines)
print(f"Stats saved to {STATS_PATH}\n")

# Print clean table
print("SUMMARY TABLE (paste into report):")
print("-"*75)
tdf = pd.DataFrame(results_table)
print(tdf.to_string(index=False))
print("-"*75)

# =====================================================================
# BOX PLOTS
# =====================================================================
COLORS = {"Normal": "#4A90D9", "Papilledema": "#E05C5C"}

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Vascular Biomarkers: Normal vs Papilledema\n(Bangladesh External Dataset)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, (col, label) in zip(axes, biomarkers.items()):
    n_vals = normal[col].dropna().values
    p_vals = papil[col].dropna().values

    bp = ax.boxplot(
        [n_vals, p_vals],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    bp["boxes"][0].set_facecolor(COLORS["Normal"])
    bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_facecolor(COLORS["Papilledema"])
    bp["boxes"][1].set_alpha(0.85)

    # Overlay individual points
    for i, vals in enumerate([n_vals, p_vals], start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        color  = list(COLORS.values())[i-1]
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   alpha=0.45, s=18, color=color, zorder=3)

    # Significance bracket
    _, pvalue = mannwhitneyu(n_vals, p_vals, alternative="two-sided")
    sig = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
    y_max = max(np.nanmax(n_vals), np.nanmax(p_vals))
    y_pad = (y_max - min(np.nanmin(n_vals), np.nanmin(p_vals))) * 0.08
    ax.plot([1, 1, 2, 2], [y_max + y_pad, y_max + y_pad*1.5,
                            y_max + y_pad*1.5, y_max + y_pad],
            color="black", linewidth=1)
    ax.text(1.5, y_max + y_pad*1.7, sig, ha="center", va="bottom", fontsize=11)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Normal", "Papilledema"], fontsize=10)
    ax.set_ylabel(label, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)

patches = [mpatches.Patch(color=COLORS["Normal"],      label="Normal"),
           mpatches.Patch(color=COLORS["Papilledema"], label="Papilledema")]
fig.legend(handles=patches, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {PLOT_PATH}")
plt.show()