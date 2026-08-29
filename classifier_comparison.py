# classifier_comparison.py
# Run in your Retinal-Vessel-Segmentation folder AFTER merge_features.py.
#
# Trains and compares:
#   Model A — Random Forest       (black-box, disc features only)
#   Model B — Logistic Regression (explainable, disc features only)
#   Model C — Hybrid Random Forest (disc features + CNN confidence score)
#
# Outputs:
#   classifier_results.txt    — full metrics table
#   roc_comparison.png        — ROC curves for all 3 models
#   feature_importance.png    — Random Forest feature importances
#   lr_coefficients.png       — Logistic Regression coefficients
#   confusion_matrices.png    — side-by-side confusion matrices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
CSV_PATH    = "combined_features.csv"
N_SPLITS    = 5       # stratified k-fold
RANDOM_SEED = 42

DISC_FEATURES = [
    "disc_brightness",
    "disc_redness",
    "disc_green_intensity",
    "disc_margin_sharpness",
    "disc_area_bright_frac",
]
CNN_FEATURE   = "cnn_prob_papil"
ALL_FEATURES  = DISC_FEATURES + [CNN_FEATURE]

# =====================================================================
# LOAD DATA
# =====================================================================
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows — classes: {df['class'].value_counts().to_dict()}\n")

le = LabelEncoder()
y  = le.fit_transform(df["class"])   # Normal=0, Papilledema=1
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

X_disc   = df[DISC_FEATURES].values
X_hybrid = df[ALL_FEATURES].values

# =====================================================================
# MODEL DEFINITIONS
# =====================================================================
models_config = {
    "Random Forest\n(disc features)": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_leaf=3, random_state=RANDOM_SEED,
                class_weight="balanced"))
        ]),
        "X":     X_disc,
        "color": "#4A90D9",
        "short": "RF",
    },
    "Logistic Regression\n(explainable)": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=1000,
                random_state=RANDOM_SEED, class_weight="balanced"))
        ]),
        "X":     X_disc,
        "color": "#E05C5C",
        "short": "LR",
    },
    "Hybrid Random Forest\n(disc + CNN score)": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_leaf=3, random_state=RANDOM_SEED,
                class_weight="balanced"))
        ]),
        "X":     X_hybrid,
        "color": "#27AE60",
        "short": "Hybrid RF",
    },
}

# =====================================================================
# CROSS-VALIDATION
# =====================================================================
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

scoring = ["f1", "precision", "recall", "roc_auc", "accuracy"]

print(f"{'='*65}")
print(f"{'Model':<35} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Acc':>6}")
print(f"{'='*65}")

cv_results = {}
lines      = ["Model Comparison — 5-Fold Stratified Cross-Validation\n",
              "="*65 + "\n"]

for name, cfg in models_config.items():
    scores = cross_validate(cfg["pipeline"], cfg["X"], y,
                            cv=cv, scoring=scoring, return_train_score=False)
    cv_results[name] = scores

    f1   = scores["test_f1"].mean()
    prec = scores["test_precision"].mean()
    rec  = scores["test_recall"].mean()
    auc  = scores["test_roc_auc"].mean()
    acc  = scores["test_accuracy"].mean()

    f1_std  = scores["test_f1"].std()
    auc_std = scores["test_roc_auc"].std()

    short = cfg["short"]
    row   = (f"{short:<35} {f1:.4f} {prec:.4f} {rec:.4f} {auc:.4f} {acc:.4f}")
    print(row)
    lines.append(row + "\n")

    print(f"  F1  = {f1:.4f} ± {f1_std:.4f}")
    print(f"  AUC = {auc:.4f} ± {auc_std:.4f}\n")

print(f"{'='*65}")

# =====================================================================
# FULL FIT on all data — for plots
# =====================================================================
fitted = {}
for name, cfg in models_config.items():
    pipe = cfg["pipeline"]
    pipe.fit(cfg["X"], y)
    fitted[name] = pipe

# =====================================================================
# ROC CURVES
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],"k--", linewidth=0.8, label="Random chance")

for name, cfg in models_config.items():
    pipe   = fitted[name]
    y_prob = pipe.predict_proba(cfg["X"])[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_val = roc_auc_score(y, y_prob)
    short   = cfg["short"]
    ax.plot(fpr, tpr, color=cfg["color"], linewidth=2,
            label=f"{short}  (AUC = {auc_val:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — Normal vs Papilledema", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=150)
print("Saved: roc_comparison.png")
plt.close()

# =====================================================================
# FEATURE IMPORTANCE — Random Forest
# =====================================================================
rf_pipe       = fitted["Random Forest\n(disc features)"]
rf_clf        = rf_pipe.named_steps["clf"]
importances   = rf_clf.feature_importances_
feat_names    = [f.replace("disc_", "").replace("_", " ").title()
                 for f in DISC_FEATURES]
sorted_idx    = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(importances)),
              importances[sorted_idx],
              color="#4A90D9", alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feat_names[i] for i in sorted_idx], rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Feature Importance (Gini)", fontsize=10)
ax.set_title("Random Forest — Feature Importances", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved: feature_importance.png")
plt.close()

# =====================================================================
# LR COEFFICIENTS — Logistic Regression (explainability plot)
# =====================================================================
lr_pipe   = fitted["Logistic Regression\n(explainable)"]
lr_clf    = lr_pipe.named_steps["clf"]
coefs     = lr_clf.coef_[0]
colors_lr = ["#E05C5C" if c > 0 else "#4A90D9" for c in coefs]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(feat_names, coefs, color=colors_lr, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (positive → higher = more Papilledema)", fontsize=9)
ax.set_title("Logistic Regression — Feature Coefficients\n"
             "(red = increases Papilledema risk, blue = decreases it)",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("lr_coefficients.png", dpi=150)
print("Saved: lr_coefficients.png")
plt.close()

# =====================================================================
# CONFUSION MATRICES — side by side (3 models)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Confusion Matrices (fit on full dataset)",
             fontsize=12, fontweight="bold")

display_names = ["RF\n(disc)", "LR\n(explainable)", "Hybrid RF\n(disc+CNN)"]
for ax, (name, cfg), dname in zip(axes, models_config.items(), display_names):
    y_pred = fitted[name].predict(cfg["X"])
    cm     = confusion_matrix(y, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Papilledema"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(dname, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("Saved: confusion_matrices.png")
plt.close()

# =====================================================================
# SAVE RESULTS TEXT
# =====================================================================
lines.append("\nFeature Importance (Random Forest):\n")
for i in sorted_idx:
    lines.append(f"  {feat_names[i]:<30}: {importances[i]:.4f}\n")

lines.append("\nLogistic Regression Coefficients:\n")
for name, coef in zip(feat_names, coefs):
    direction = "→ higher = more Papilledema" if coef > 0 else "→ lower = more Papilledema"
    lines.append(f"  {name:<30}: {coef:+.4f}  {direction}\n")

with open("classifier_results.txt", "w") as f:
    f.writelines(lines)
print("Saved: classifier_results.txt")

# =====================================================================
# FINAL SUMMARY PRINT
# =====================================================================
print(f"\n{'='*65}")
print("FINAL SUMMARY")
print(f"{'='*65}")
for name, cfg in models_config.items():
    scores = cv_results[name]
    short  = cfg["short"]
    print(f"\n{short}:")
    print(f"  F1   = {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")
    print(f"  AUC  = {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"  Acc  = {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")

print(f"\n{'='*65}")
print("KEY QUESTION: Is there a substantial F1 difference between RF and LR?")
rf_f1 = cv_results["Random Forest\n(disc features)"]["test_f1"].mean()
lr_f1 = cv_results["Logistic Regression\n(explainable)"]["test_f1"].mean()
diff  = abs(rf_f1 - lr_f1)
print(f"  RF F1:  {rf_f1:.4f}")
print(f"  LR F1:  {lr_f1:.4f}")
print(f"  Diff:   {diff:.4f}")
if diff < 0.02:
    print("  → Negligible difference (<0.02). Use LR — same performance, fully explainable.")
elif diff < 0.05:
    print("  → Small difference (0.02–0.05). LR is still preferred for explainability.")
else:
    print("  → Substantial difference (>0.05). RF meaningfully outperforms LR.")
print(f"{'='*65}")
