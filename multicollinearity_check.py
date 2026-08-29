# multicollinearity_check.py
# Runs RF and LR with three feature sets:
#   (A) all 5 features (baseline)
#   (B) drop brightness, keep green intensity
#   (C) drop green intensity, keep brightness
# Reports F1 and AUC for each combination.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

CSV_PATH    = "combined_features.csv"
RANDOM_SEED = 42
N_SPLITS    = 5

# =====================================================================
# LOAD
# =====================================================================
df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
y  = le.fit_transform(df["class"])
print(f"Loaded {len(df)} rows — {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# =====================================================================
# STEP 1 — CORRELATION CHECK
# =====================================================================
b  = df["disc_brightness"].values
gi = df["disc_green_intensity"].values

pearson_r,  pearson_p  = pearsonr(b, gi)
spearman_r, spearman_p = spearmanr(b, gi)

print("="*55)
print("CORRELATION: disc_brightness vs disc_green_intensity")
print("="*55)
print(f"  Pearson  r = {pearson_r:.4f}  (p={pearson_p:.2e})")
print(f"  Spearman r = {spearman_r:.4f}  (p={spearman_p:.2e})")

if abs(pearson_r) > 0.85:
    print(f"  → STRONG multicollinearity confirmed (r>{0.85})")
elif abs(pearson_r) > 0.70:
    print(f"  → Moderate multicollinearity (r>{0.70})")
else:
    print(f"  → Low multicollinearity — may not need removal")
print()

# =====================================================================
# STEP 2 — SCATTER PLOT of the two correlated features
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 5))
colors  = ["#4A90D9" if c == 0 else "#E05C5C" for c in y]
ax.scatter(b, gi, c=colors, alpha=0.5, s=20)
ax.set_xlabel("Disc Brightness", fontsize=11)
ax.set_ylabel("Green Channel Intensity", fontsize=11)
ax.set_title(f"Brightness vs Green Intensity\n"
             f"Pearson r={pearson_r:.3f} — confirms multicollinearity",
             fontsize=11, fontweight="bold")

import matplotlib.patches as mpatches
legend = [mpatches.Patch(color="#4A90D9", label="Normal"),
          mpatches.Patch(color="#E05C5C", label="Papilledema")]
ax.legend(handles=legend, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("collinearity_scatter.png", dpi=150)
print("Saved: collinearity_scatter.png\n")
plt.close()

# =====================================================================
# STEP 3 — FEATURE SETS
# =====================================================================
feature_sets = {
    "All 5 (baseline)": [
        "disc_brightness", "disc_redness", "disc_green_intensity",
        "disc_margin_sharpness", "disc_area_bright_frac"
    ],
    "Drop brightness\n(keep green)": [
        "disc_redness", "disc_green_intensity",
        "disc_margin_sharpness", "disc_area_bright_frac"
    ],
    "Drop green intensity\n(keep brightness)": [
        "disc_brightness", "disc_redness",
        "disc_margin_sharpness", "disc_area_bright_frac"
    ],
}

# =====================================================================
# STEP 4 — RUN BOTH MODELS ON EACH FEATURE SET
# =====================================================================
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
scoring = ["f1", "roc_auc", "precision", "recall"]

all_results = []

print("="*75)
print(f"{'Feature Set':<35} {'Model':<22} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6}")
print("="*75)

for fs_name, features in feature_sets.items():
    X = df[features].values

    for model_name, clf in [
        ("Random Forest",        RandomForestClassifier(
                                     n_estimators=200, max_depth=8,
                                     min_samples_leaf=3, random_state=RANDOM_SEED,
                                     class_weight="balanced")),
        ("Logistic Regression",  LogisticRegression(
                                     C=1.0, max_iter=1000,
                                     random_state=RANDOM_SEED, class_weight="balanced")),
    ]:
        pipe   = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

        f1   = scores["test_f1"].mean()
        auc  = scores["test_roc_auc"].mean()
        prec = scores["test_precision"].mean()
        rec  = scores["test_recall"].mean()
        f1_std  = scores["test_f1"].std()
        auc_std = scores["test_roc_auc"].std()

        short_fs = fs_name.replace("\n", " ")
        print(f"{short_fs:<35} {model_name:<22} {f1:.4f} {auc:.4f} {prec:.4f} {rec:.4f}")

        all_results.append({
            "feature_set":  short_fs,
            "model":        model_name,
            "n_features":   len(features),
            "features":     ", ".join(f.replace("disc_","") for f in features),
            "F1":           round(f1, 4),
            "F1_std":       round(f1_std, 4),
            "AUC":          round(auc, 4),
            "AUC_std":      round(auc_std, 4),
            "Precision":    round(prec, 4),
            "Recall":       round(rec, 4),
        })

    print("-"*75)

# =====================================================================
# STEP 5 — LR COEFFICIENTS for each feature set (explainability check)
# =====================================================================
print("\n" + "="*55)
print("LOGISTIC REGRESSION COEFFICIENTS per feature set")
print("(check: brightness/green coeff signs should make sense)")
print("="*55)

for fs_name, features in feature_sets.items():
    X    = df[features].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000,
                                      random_state=RANDOM_SEED,
                                      class_weight="balanced"))
    ])
    pipe.fit(X, y)
    coefs = pipe.named_steps["clf"].coef_[0]
    short_names = [f.replace("disc_","").replace("_"," ") for f in features]

    print(f"\n{fs_name.replace(chr(10),' ')}:")
    for fname, coef in zip(short_names, coefs):
        direction = "→ more Papilledema" if coef > 0 else "→ more Normal"
        expected  = ""
        if "brightness" in fname:
            expected = " ✓" if coef < 0 else " ✗ (should be negative)"
        if "green" in fname:
            expected = " ✓" if coef < 0 else " ✗ (should be negative)"
        if "redness" in fname:
            expected = " ✓" if coef > 0 else " ✗ (should be positive)"
        if "sharpness" in fname:
            expected = " ✓" if coef < 0 else " ✗ (should be negative)"
        if "bright frac" in fname or "bright_frac" in fname:
            expected = " ✓" if coef < 0 else " ✗ (should be negative)"
        print(f"  {fname:<25}: {coef:+.4f}  {direction}{expected}")

# =====================================================================
# STEP 6 — BAR CHART COMPARISON
# =====================================================================
results_df = pd.DataFrame(all_results)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Multicollinearity Check: Effect of Removing Brightness or Green Intensity",
             fontsize=12, fontweight="bold")

metrics   = ["F1", "AUC"]
titles    = ["F1 Score (5-fold CV)", "AUC (5-fold CV)"]
bar_colors = {"Random Forest": "#4A90D9", "Logistic Regression": "#E05C5C"}

fs_labels  = ["All 5 (baseline)",
              "Drop brightness (keep green)",
              "Drop green intensity (keep brightness)"]
x          = np.arange(len(fs_labels))
width      = 0.35

for ax, metric, title in zip(axes, metrics, titles):
    for i, (model_name, color) in enumerate(bar_colors.items()):
        vals = [
            results_df[(results_df["feature_set"] == fs.replace("\n"," ")) &
                       (results_df["model"] == model_name)][metric].values[0]
            for fs in fs_labels
        ]
        stds = [
            results_df[(results_df["feature_set"] == fs.replace("\n"," ")) &
                       (results_df["model"] == model_name)][f"{metric}_std"].values[0]
            for fs in fs_labels
        ]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model_name,
                        color=color, alpha=0.85, yerr=stds,
                        capsize=4, error_kw=dict(linewidth=1))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["All 5\n(baseline)", "Drop\nBrightness", "Drop\nGreen Int."],
                       fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("multicollinearity_comparison.png", dpi=150)
print("\n\nSaved: multicollinearity_comparison.png")
plt.close()

# =====================================================================
# STEP 7 — CONCLUSION
# =====================================================================
print("\n" + "="*55)
print("CONCLUSION")
print("="*55)

rf_base    = results_df[(results_df["feature_set"]=="All 5 (baseline)") &
                         (results_df["model"]=="Random Forest")].iloc[0]
rf_drop_b  = results_df[(results_df["feature_set"]=="Drop brightness (keep green)") &
                         (results_df["model"]=="Random Forest")].iloc[0]
rf_drop_g  = results_df[(results_df["feature_set"]=="Drop green intensity (keep brightness)") &
                         (results_df["model"]=="Random Forest")].iloc[0]

lr_base    = results_df[(results_df["feature_set"]=="All 5 (baseline)") &
                         (results_df["model"]=="Logistic Regression")].iloc[0]
lr_drop_b  = results_df[(results_df["feature_set"]=="Drop brightness (keep green)") &
                         (results_df["model"]=="Logistic Regression")].iloc[0]
lr_drop_g  = results_df[(results_df["feature_set"]=="Drop green intensity (keep brightness)") &
                         (results_df["model"]=="Logistic Regression")].iloc[0]

print(f"\nRandom Forest:")
print(f"  Baseline (all 5)      : F1={rf_base['F1']:.4f}  AUC={rf_base['AUC']:.4f}")
print(f"  Drop brightness       : F1={rf_drop_b['F1']:.4f}  AUC={rf_drop_b['AUC']:.4f}  "
      f"({'↑' if rf_drop_b['F1'] > rf_base['F1'] else '↓'} F1)")
print(f"  Drop green intensity  : F1={rf_drop_g['F1']:.4f}  AUC={rf_drop_g['AUC']:.4f}  "
      f"({'↑' if rf_drop_g['F1'] > rf_base['F1'] else '↓'} F1)")

print(f"\nLogistic Regression:")
print(f"  Baseline (all 5)      : F1={lr_base['F1']:.4f}  AUC={lr_base['AUC']:.4f}")
print(f"  Drop brightness       : F1={lr_drop_b['F1']:.4f}  AUC={lr_drop_b['AUC']:.4f}  "
      f"({'↑' if lr_drop_b['F1'] > lr_base['F1'] else '↓'} F1)")
print(f"  Drop green intensity  : F1={lr_drop_g['F1']:.4f}  AUC={lr_drop_g['AUC']:.4f}  "
      f"({'↑' if lr_drop_g['F1'] > lr_base['F1'] else '↓'} F1)")

# Recommendation
best_lr = max([
    ("all 5", lr_base["AUC"]),
    ("drop brightness (keep green)", lr_drop_b["AUC"]),
    ("drop green intensity (keep brightness)", lr_drop_g["AUC"])
], key=lambda x: x[1])

print(f"\nRecommendation for LR (AUC-based): {best_lr[0]}  (AUC={best_lr[1]:.4f})")
print("Check coefficient signs above — the better feature set will show")
print("all signs matching clinical expectations (✓ markers).")
print("\nSaved all outputs. Share multicollinearity_comparison.png with professor.")
