# bangladesh_biomarkers.py
#
# Run this from inside your Retinal-Vessel-Segmentation folder.
# Segments vessels on all Bangladesh Normal/Papilledema images
# using your trained U-Net, computes 3 biomarkers per image,
# and saves bangladesh_biomarkers.csv + per-class mask PNGs.
#
# U-Net input:  (None, 48, 48, 1)
# U-Net output: (None, 2304, 2)  → reshaped to (N, 48, 48, 2)

import os
import sys
import glob
import numpy as np
import imageio.v2 as imageio
import pandas as pd
from PIL import Image

from skimage.morphology import skeletonize
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import convolve

sys.path.insert(0, './lib/')
from pre_processing import my_PreProc
from keras.models import load_model

# =====================================================================
# CONFIG
# =====================================================================
MODEL_PATH      = "test_drive/test_drive_best_model.keras"
DATA_ROOT       = "bangladesh_data"          # Normal/ and Papilledema/ inside
MASK_OUTPUT     = "bangladesh_masks"
RESULTS_CSV     = "bangladesh_biomarkers.csv"
PATCH_H         = 48
PATCH_W         = 48
THRESHOLD       = 0.4
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(os.path.join(MASK_OUTPUT, "Normal"),      exist_ok=True)
os.makedirs(os.path.join(MASK_OUTPUT, "Papilledema"), exist_ok=True)

# =====================================================================
# LOAD MODEL ONCE
# =====================================================================
print("Loading U-Net model...")
model = load_model(MODEL_PATH)
print(f"  Input:  {model.input_shape}")
print(f"  Output: {model.output_shape}")
print("  Model ready.\n")

# =====================================================================
# PREPROCESSING  (matches training exactly)
# =====================================================================
def preprocess(image_path):
    """
    Load image → my_PreProc → return (H, W, 1) float32 array
    and original (H, W) shape.
    """
    img    = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)              # (H, W, 3)

    # my_PreProc expects (N, 3, H, W) channels-first
    img_cf = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
    proc   = my_PreProc(img_cf)                                  # (1, 1, H, W)

    h, w        = proc.shape[2], proc.shape[3]
    img_array   = proc[0, 0, :, :][:, :, np.newaxis]    # (H, W, 1)
    return img_array, h, w

# =====================================================================
# VESSEL SEGMENTATION
# =====================================================================
def segment_vessels(image_path):
    """
    Run U-Net patch inference → binary mask (uint8, 0/1).
    """
    img_array, h, w = preprocess(image_path)

    # Pad to multiple of patch size
    padded_h = int(np.ceil(h / PATCH_H) * PATCH_H)
    padded_w = int(np.ceil(w / PATCH_W) * PATCH_W)

    padded = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
    padded[:h, :w, :] = img_array

    # Extract patches
    patches, coords = [], []
    for y in range(0, padded_h, PATCH_H):
        for x in range(0, padded_w, PATCH_W):
            patches.append(padded[y:y+PATCH_H, x:x+PATCH_W, :])
            coords.append((y, x))

    patches = np.array(patches)                          # (N, 48, 48, 1)

    # Predict
    preds = model.predict(patches, verbose=0)            # (N, 2304, 2)

    # Reshape to spatial
    preds = preds.reshape(-1, PATCH_H, PATCH_W, 2)      # (N, 48, 48, 2)
    vessel_maps = preds[:, :, :, 1]                      # vessel channel

    # Reconstruct
    pred_full = np.zeros((padded_h, padded_w), dtype=np.float32)
    for (y, x), vmap in zip(coords, vessel_maps):
        pred_full[y:y+PATCH_H, x:x+PATCH_W] = vmap

    pred_full   = pred_full[:h, :w]
    binary_mask = (pred_full >= THRESHOLD).astype(np.uint8)
    return binary_mask

# =====================================================================
# BIOMARKER FUNCTIONS  (all bugs fixed)
# =====================================================================
def vessel_density(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.nan
    roi = mask[ys.min():ys.max(), xs.min():xs.max()]
    return float(np.sum(roi) / roi.size) if roi.size > 0 else np.nan


def mean_tortuosity(skeleton):
    kernel        = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbor_count = convolve(skeleton.astype(int), kernel, mode="constant")
    pruned        = skeleton.copy()
    pruned[neighbor_count >= 13] = 0

    labeled = sk_label(pruned)
    torts   = []
    for region in regionprops(labeled):
        coords = region.coords
        if len(coords) < 5:
            continue
        path_len = float(np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))))
        euclid   = float(np.linalg.norm(coords[0] - coords[-1]))
        if euclid > 0:
            torts.append(path_len / euclid)
    return float(np.mean(torts)) if torts else np.nan


def fractal_dimension(Z):
    Z    = Z.astype(bool)
    ys, xs = np.where(Z)
    if len(xs) == 0:
        return np.nan
    Z = Z[ys.min():ys.max(), xs.min():xs.max()]
    if min(Z.shape) < 2:
        return np.nan

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero(S)

    p = min(Z.shape)
    n = int(2 ** np.floor(np.log2(p)))
    valid_s, valid_c = [], []
    for s in 2 ** np.arange(int(np.log2(n)), 1, -1):
        c = boxcount(Z, int(s))
        if c > 0:
            valid_s.append(s)
            valid_c.append(c)
    if len(valid_s) < 2:
        return np.nan
    return float(-np.polyfit(np.log(valid_s), np.log(valid_c), 1)[0])

# =====================================================================
# MAIN LOOP
# =====================================================================
classes = {"Normal": "Normal", "Papilledema": "Papilledema"}
results = []

for class_folder, class_label in classes.items():
    folder = os.path.join(DATA_ROOT, class_folder)
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found — {folder}")
        continue

    image_paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]

    print(f"{'='*60}")
    print(f"Class: {class_label}  ({len(image_paths)} images)")
    print(f"{'='*60}")

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"  [{idx+1:>3}/{len(image_paths)}] {filename}", end=" ... ", flush=True)

        try:
            mask     = segment_vessels(img_path)
            skeleton = skeletonize(mask > 0)

            d  = vessel_density(mask)
            t  = mean_tortuosity(skeleton)
            fd = fractal_dimension(skeleton)

            # Save mask
            stem      = os.path.splitext(filename)[0]
            mask_path = os.path.join(MASK_OUTPUT, class_label, stem + ".png")
            imageio.imwrite(mask_path, mask * 255)

            results.append({
                "image":            filename,
                "class":            class_label,
                "vessel_density":   d,
                "tortuosity":       t,
                "fractal_dimension": fd,
            })
            print(f"density={d:.3f}  tort={t:.3f}  fd={fd:.3f}")

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({
                "image": filename, "class": class_label,
                "vessel_density": np.nan,
                "tortuosity": np.nan,
                "fractal_dimension": np.nan,
            })

# =====================================================================
# SAVE CSV
# =====================================================================
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*60}")
print(f"Saved: {RESULTS_CSV}  ({len(df)} rows)")
print(f"{'='*60}\n")

for cls in ["Normal", "Papilledema"]:
    sub = df[df["class"] == cls]
    print(f"{cls} (n={len(sub)}):")
    for col in ["vessel_density", "tortuosity", "fractal_dimension"]:
        vals = sub[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<22}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"nan={sub[col].isna().sum()}")
    print()

print("Next step: run  python3 biomarker_analysis.py")