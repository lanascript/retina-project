# merge_features.py
# Run in your Retinal-Vessel-Segmentation folder.
# Merges disc_features.csv + cnn_scores.csv → combined_features.csv
# Also does a quick sanity check on the merge.

import pandas as pd
import numpy as np

DISC_CSV   = "disc_features.csv"
CNN_CSV    = "cnn_scores.csv"
OUTPUT_CSV = "combined_features.csv"

# =====================================================================
# LOAD
# =====================================================================
disc = pd.read_csv(DISC_CSV)
cnn  = pd.read_csv(CNN_CSV)

print(f"Disc features : {disc.shape}  — classes: {disc['class'].value_counts().to_dict()}")
print(f"CNN scores    : {cnn.shape}   — classes: {cnn['class'].value_counts().to_dict()}")

# =====================================================================
# MERGE on image filename + class
# =====================================================================
merged = pd.merge(disc, cnn[["image", "class", "cnn_prob_normal", "cnn_prob_papil",
                               "cnn_prob_pseudo", "cnn_pred"]],
                  on=["image", "class"], how="inner")

print(f"\nAfter merge   : {merged.shape}")
print(f"Classes       : {merged['class'].value_counts().to_dict()}")

# Check for NaNs
nan_counts = merged.isnull().sum()
if nan_counts.any():
    print(f"\nWARNING — NaN values found:\n{nan_counts[nan_counts > 0]}")
else:
    print("No NaN values — clean merge.")

# Drop rows with any NaN in feature columns
feature_cols = ["disc_brightness", "disc_redness", "disc_green_intensity",
                "disc_margin_sharpness", "disc_area_bright_frac",
                "cnn_prob_papil"]
before = len(merged)
merged = merged.dropna(subset=feature_cols)
after  = len(merged)
if before != after:
    print(f"Dropped {before - after} rows with NaN in feature columns.")

merged.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}  ({len(merged)} rows)")
print("\nColumns in combined file:")
for c in merged.columns:
    print(f"  {c}")
