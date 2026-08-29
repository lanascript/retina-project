# pca_classifier.py
# PCA-based decorrelation of disc features, then RF + LR retest.
# Run in Retinal-Vessel-Segmentation folder after combined_features.csv exists.
#
# Outputs:
#   pca_scree.png              — variance explained per component
#   pca_scatter.png            — PCA space coloured by class
#   pca_classifier_results.txt — F1 and AUC for all models
#   pca_comparison.png         — bar chart comparing all approaches

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

CSV_PATH    = "combined_features.csv"
RANDOM_SEED = 42
N_SPLITS    = 5

DISC_FEATURES = [
    "disc_brightness", "disc_redness", "disc_green_intensity",
    "disc_margin_sharpness", "disc_area_bright_frac",
]

# =====================================================================
# LOAD
# =====================================================================
df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
y  = le.fit_transform(df["class"])
X  = df[DISC_FEATURES].values
print(f"Loaded {len(df)} rows\n")

# =====================================================================
# STEP 1 — FIT PCA on standardised features
# =====================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)

print("PCA Explained Variance Ratio:")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {v:.4f}  ({v*100:.1f}%)")
print(f"  Cumulative (PC1+PC2): {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
print(f"  Cumulative (all 5):   {pca.explained_variance_ratio_.sum()*100:.1f}%\n")

# =====================================================================
# STEP 2 — SCREE PLOT
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.bar(range(1, 6), pca.explained_variance_ratio_ * 100,
       color="#4A90D9", alpha=0.85, edgecolor="white")
ax.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_) * 100,
        "o--", color="#E05C5C", linewidth=1.5, label="Cumulative %")
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"PC{i}" for i in range(1, 6)])
ax.set_ylabel("Variance Explained (%)", fontsize=10)
ax.set_title("PCA Scree Plot", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# =====================================================================
# STEP 3 — PCA SCATTER (PC1 vs PC2)
# =====================================================================
ax = axes[1]
colors_map = {0: "#4A90D9", 1: "#E05C5C"}
for cls, label in zip([0, 1], ["Normal", "Papilledema"]):
    idx = y == cls
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
               c=colors_map[cls], alpha=0.5, s=18, label=label)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax.set_title("PCA Space — Normal vs Papilledema", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150)
print("Saved: pca_scree.png")
plt.close()

# =====================================================================
# STEP 4 — PCA LOADINGS (which original features dominate each PC)
# =====================================================================
loadings = pd.DataFrame(
    pca.components_.T,
    index=[f.replace("disc_","").replace("_"," ") for f in DISC_FEATURES],
    columns=[f"PC{i+1}" for i in range(5)]
)
print("PCA Loadings (contribution of each feature to each PC):")
print(loadings.round(3).to_string())
print()

# =====================================================================
# STEP 5 — CROSS-VALIDATION with PCA components
# Choose number of components: test 2, 3, 4, 5
# =====================================================================
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
scoring = ["f1", "roc_auc"]

all_results = []

print("="*70)
print(f"{'Config':<35} {'Model':<22} {'F1':>7} {'AUC':>7}")
print("="*70)

# Reference: original 5 features (no PCA)
for model_name, clf in [
    ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    min_samples_leaf=3,
                                                    random_state=RANDOM_SEED,
                                                    class_weight="balanced")),
    ("Logistic Regression", LogisticRegression(C=1.0, max_iter=1000,
                                                random_state=RANDOM_SEED,
                                                class_weight="balanced")),
]:
    pipe   = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    f1     = scores["test_f1"].mean()
    auc    = scores["test_roc_auc"].mean()
    config = "Original 5 features"
    print(f"{config:<35} {model_name:<22} {f1:.4f} {auc:.4f}")
    all_results.append({"config": config, "model": model_name,
                         "n_components": 5, "F1": f1, "AUC": auc,
                         "F1_std": scores["test_f1"].std(),
                         "AUC_std": scores["test_roc_auc"].std()})

print("-"*70)

# PCA variants
for n_comp in [2, 3, 4, 5]:
    X_pc = X_pca[:, :n_comp]
    var_explained = pca.explained_variance_ratio_[:n_comp].sum() * 100
    config = f"PCA {n_comp} components ({var_explained:.0f}% var)"

    for model_name, clf in [
        ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                        min_samples_leaf=3,
                                                        random_state=RANDOM_SEED,
                                                        class_weight="balanced")),
        ("Logistic Regression", LogisticRegression(C=1.0, max_iter=1000,
                                                    random_state=RANDOM_SEED,
                                                    class_weight="balanced")),
    ]:
        # Note: data already scaled, no additional scaler needed for PCA components
        pipe   = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_validate(pipe, X_pc, y, cv=cv, scoring=scoring)
        f1     = scores["test_f1"].mean()
        auc    = scores["test_roc_auc"].mean()
        print(f"{config:<35} {model_name:<22} {f1:.4f} {auc:.4f}")
        all_results.append({"config": config, "model": model_name,
                             "n_components": n_comp, "F1": f1, "AUC": auc,
                             "F1_std": scores["test_f1"].std(),
                             "AUC_std": scores["test_roc_auc"].std()})

    print("-"*70)

# =====================================================================
# STEP 6 — LR COEFFICIENTS after PCA (should be stable)
# =====================================================================
print("\nLR Coefficients after PCA (PC1-PC5):")
print("(Stable coefficients = multicollinearity resolved)\n")

for n_comp in [2, 3, 5]:
    X_pc   = X_pca[:, :n_comp]
    scaler2 = StandardScaler()
    X_pc_s = scaler2.fit_transform(X_pc)
    lr     = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED,
                                 class_weight="balanced")
    lr.fit(X_pc_s, y)
    coefs  = lr.coef_[0]
    print(f"  PCA {n_comp} components: " +
          "  ".join([f"PC{i+1}={c:+.3f}" for i, c in enumerate(coefs)]))

# =====================================================================
# STEP 7 — COMPARISON BAR CHART
# =====================================================================
results_df = pd.DataFrame(all_results)
rf_results = results_df[results_df["model"] == "Random Forest"]
lr_results = results_df[results_df["model"] == "Logistic Regression"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA vs Original Features — RF and LR Performance",
             fontsize=12, fontweight="bold")

x_labels = ["Orig.\n5 feat", "PCA\n2 PC", "PCA\n3 PC", "PCA\n4 PC", "PCA\n5 PC"]
x        = np.arange(len(x_labels))
width    = 0.35

for ax, metric, title in zip(axes, ["F1", "AUC"], ["F1 Score (5-fold CV)", "AUC (5-fold CV)"]):
    rf_vals = rf_results[metric].values
    lr_vals = lr_results[metric].values
    rf_std  = rf_results[f"{metric}_std"].values
    lr_std  = lr_results[f"{metric}_std"].values

    ax.bar(x - width/2, rf_vals, width, label="Random Forest",
           color="#4A90D9", alpha=0.85, yerr=rf_std, capsize=3,
           error_kw=dict(linewidth=1))
    ax.bar(x + width/2, lr_vals, width, label="Logistic Regression",
           color="#E05C5C", alpha=0.85, yerr=lr_std, capsize=3,
           error_kw=dict(linewidth=1))

    for i, (rv, lv) in enumerate(zip(rf_vals, lr_vals)):
        ax.text(i - width/2, rv + 0.005, f"{rv:.3f}", ha="center",
                va="bottom", fontsize=7.5)
        ax.text(i + width/2, lv + 0.005, f"{lv:.3f}", ha="center",
                va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("pca_comparison.png", dpi=150)
print("\nSaved: pca_comparison.png")
plt.close()

# =====================================================================
# STEP 8 — SAVE RESULTS
# =====================================================================
lines = ["PCA Classifier Results\n", "="*70 + "\n\n"]
lines.append("PCA Explained Variance:\n")
for i, v in enumerate(pca.explained_variance_ratio_):
    lines.append(f"  PC{i+1}: {v*100:.1f}%  (cumulative: "
                 f"{pca.explained_variance_ratio_[:i+1].sum()*100:.1f}%)\n")
lines.append("\nPCA Loadings:\n")
lines.append(loadings.round(3).to_string() + "\n\n")
lines.append("Cross-Validation Results:\n")
for r in all_results:
    lines.append(f"  {r['config']:<35} {r['model']:<22} "
                 f"F1={r['F1']:.4f}±{r['F1_std']:.4f}  "
                 f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}\n")

with open("pca_classifier_results.txt", "w") as f:
    f.writelines(lines)
print("Saved: pca_classifier_results.txt")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "="*70)
print("SUMMARY — Best configuration per model:")
best_rf = rf_results.loc[rf_results["F1"].idxmax()]
best_lr = lr_results.loc[lr_results["F1"].idxmax()]
print(f"  RF best:  {best_rf['config']}  F1={best_rf['F1']:.4f}  AUC={best_rf['AUC']:.4f}")
print(f"  LR best:  {best_lr['config']}  F1={best_lr['F1']:.4f}  AUC={best_lr['AUC']:.4f}")
print("="*70)
print("\nDid PCA improve LR?")
lr_orig = lr_results[lr_results["config"]=="Original 5 features"]["F1"].values[0]
lr_best = best_lr["F1"]
if lr_best > lr_orig + 0.01:
    print(f"  YES — LR F1 improved from {lr_orig:.4f} to {lr_best:.4f} (+{lr_best-lr_orig:.4f})")
    print("  Multicollinearity was causing LR to underperform. PCA resolved it.")
else:
    print(f"  MARGINAL — LR F1 changed from {lr_orig:.4f} to {lr_best:.4f}")
    print("  Features may not be linearly separable enough for LR regardless.")