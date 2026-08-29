# disc_features.py
#
# Extracts optic disc region features from Bangladesh fundus images
# and runs statistical comparison between Normal and Papilledema.
#
# Method: auto-detect disc via brightest region in green channel,
# crop a fixed window around it, compute 6 disc-relevant features.
#
# Run from inside Retinal-Vessel-Segmentation folder:
#   python3 disc_features.py

import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu
from skimage.filters import sobel

# =====================================================================
# CONFIG
# =====================================================================
DATA_ROOT   = "bangladesh_data"
RESULTS_CSV = "disc_features.csv"
PLOT_PATH   = "disc_feature_boxplots.png"
STATS_PATH  = "disc_feature_stats.txt"
DEBUG_DIR   = "disc_debug"          # saves crop previews for first 5 images
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

# Disc crop: radius as fraction of image shorter dimension
# 2004x1690 → shorter = 1690 → disc window = 1690 * 0.18 ≈ 304px radius
DISC_RADIUS_FRAC = 0.18

os.makedirs(DEBUG_DIR, exist_ok=True)

# =====================================================================
# DISC DETECTION  — brightest blob in green channel
# =====================================================================
def find_disc_centre(img_np):
    """
    img_np: (H, W, 3) uint8 RGB
    Returns (cx, cy) pixel coordinates of estimated disc centre.
    """
    green = img_np[:, :, 1].astype(np.float32)

    # Exclude black border (common in fundus images)
    mask = green > 20

    # Smooth heavily to find broad bright region
    smoothed = gaussian_filter(green * mask, sigma=min(img_np.shape[:2]) * 0.04)

    # Find brightest point
    cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    return int(cx), int(cy)


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def extract_disc_features(img_np, save_debug_path=None):
    """
    img_np: (H, W, 3) uint8 RGB
    Returns dict of 6 features.
    """
    H, W = img_np.shape[:2]
    radius = int(min(H, W) * DISC_RADIUS_FRAC)

    cx, cy = find_disc_centre(img_np)

    # Clamp crop to image bounds
    x1 = max(0, cx - radius)
    x2 = min(W, cx + radius)
    y1 = max(0, cy - radius)
    y2 = min(H, cy + radius)

    crop = img_np[y1:y2, x1:x2]   # (crop_H, crop_W, 3)

    if crop.size == 0:
        return {k: np.nan for k in [
            "disc_brightness", "disc_redness", "disc_green_intensity",
            "disc_margin_sharpness", "disc_intensity_std", "disc_area_bright_frac"
        ]}

    r = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    b = crop[:, :, 2].astype(np.float32)

    # 1. Overall brightness of disc region
    brightness = float(np.mean((r + g + b) / 3.0))

    # 2. Redness ratio — papilledema discs appear hyperaemic (redder)
    total = r + g + b + 1e-6
    redness = float(np.mean(r / total))

    # 3. Green channel intensity — vessels and disc structure
    green_intensity = float(np.mean(g))

    # 4. Margin sharpness — blurred margins = lower edge strength
    #    Use Sobel on grayscale crop
    gray_crop = np.mean(crop, axis=2).astype(np.float32) / 255.0
    edges = sobel(gray_crop)
    # Focus on the border ring of the crop (outer 25%) where disc margin sits
    border_mask = np.zeros_like(edges, dtype=bool)
    h_c, w_c = gray_crop.shape
    cy_c, cx_c = h_c // 2, w_c // 2
    for yy in range(h_c):
        for xx in range(w_c):
            d = np.sqrt((yy - cy_c)**2 + (xx - cx_c)**2)
            if d > radius * 0.65:
                border_mask[yy, xx] = True
    margin_sharpness = float(np.mean(edges[border_mask])) if border_mask.any() else np.nan

    # 5. Intensity std — swollen discs have more uniform/washed brightness
    intensity_std = float(np.std(gray_crop))

    # 6. Fraction of disc region above brightness threshold (bright area coverage)
    bright_thresh = 0.55
    bright_frac = float(np.mean(gray_crop > bright_thresh))

    # Debug: save crop with detected centre marked
    if save_debug_path is not None:
        debug_img = img_np.copy()
        cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 4)
        cv2.circle(debug_img, (cx, cy), 6,      (255, 0, 0), -1)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        # Resize for saving (images are large)
        scale = 400 / max(H, W)
        small = cv2.resize(debug_img, (int(W*scale), int(H*scale)))
        cv2.imwrite(save_debug_path, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))

    return {
        "disc_brightness":       brightness,
        "disc_redness":          redness,
        "disc_green_intensity":  green_intensity,
        "disc_margin_sharpness": margin_sharpness,
        "disc_intensity_std":    intensity_std,
        "disc_area_bright_frac": bright_frac,
    }


# =====================================================================
# MAIN LOOP
# =====================================================================
classes  = {"Normal": "Normal", "Papilledema": "Papilledema"}
results  = []
debug_count = {"Normal": 0, "Papilledema": 0}

for class_folder, class_label in classes.items():
    folder = os.path.join(DATA_ROOT, class_folder)
    if not os.path.isdir(folder):
        print(f"WARNING: {folder} not found — skipping")
        continue

    image_paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]

    print(f"\n{'='*55}")
    print(f"{class_label}  ({len(image_paths)} images)")
    print(f"{'='*55}")

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # Save debug crop for first 5 per class
            debug_path = None
            if debug_count[class_label] < 5:
                debug_path = os.path.join(
                    DEBUG_DIR, f"{class_label}_{debug_count[class_label]+1}_{filename}"
                )
                debug_count[class_label] += 1

            feats = extract_disc_features(img_np, save_debug_path=debug_path)
            feats["image"] = filename
            feats["class"] = class_label
            results.append(feats)

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1:>3}/{len(image_paths)}] {filename}  "
                      f"bright={feats['disc_brightness']:.1f}  "
                      f"sharp={feats['disc_margin_sharpness']:.4f}")

        except Exception as e:
            import traceback
            print(f"  ERROR {filename}: {e}")
            traceback.print_exc()
            results.append({"image": filename, "class": class_label,
                             **{k: np.nan for k in [
                                 "disc_brightness","disc_redness","disc_green_intensity",
                                 "disc_margin_sharpness","disc_intensity_std","disc_area_bright_frac"
                             ]}})

# =====================================================================
# SAVE CSV
# =====================================================================
df = pd.DataFrame(results)
cols = ["image","class","disc_brightness","disc_redness","disc_green_intensity",
        "disc_margin_sharpness","disc_intensity_std","disc_area_bright_frac"]
df = df[cols]
df.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved {RESULTS_CSV}  ({len(df)} rows)")

# =====================================================================
# STATS
# =====================================================================
feature_labels = {
    "disc_brightness":       "Disc Brightness",
    "disc_redness":          "Disc Redness Ratio",
    "disc_green_intensity":  "Green Channel Intensity",
    "disc_margin_sharpness": "Margin Sharpness",
    "disc_intensity_std":    "Intensity Std Dev",
    "disc_area_bright_frac": "Bright Area Fraction",
}

normal = df[df["class"] == "Normal"]
papil  = df[df["class"] == "Papilledema"]

stats_lines = ["Disc Feature Stats — Normal vs Papilledema\n", "="*55+"\n"]
table_rows  = []

print(f"\n{'='*55}")
print("STATISTICAL RESULTS")
print(f"{'='*55}")

for col, label in feature_labels.items():
    n_vals = normal[col].dropna().values
    p_vals = papil[col].dropna().values
    if len(n_vals) < 2 or len(p_vals) < 2:
        continue

    stat, pvalue = mannwhitneyu(n_vals, p_vals, alternative="two-sided")
    sig = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"

    line = (f"{label}:\n"
            f"  Normal      : {n_vals.mean():.4f} ± {n_vals.std():.4f}\n"
            f"  Papilledema : {p_vals.mean():.4f} ± {p_vals.std():.4f}\n"
            f"  p={pvalue:.4f}  {sig}\n")
    stats_lines.append(line)
    print(line)

    table_rows.append({
        "Feature": label,
        "Normal": f"{n_vals.mean():.4f}±{n_vals.std():.4f}",
        "Papilledema": f"{p_vals.mean():.4f}±{p_vals.std():.4f}",
        "p-value": f"{pvalue:.4f}", "sig": sig
    })

with open(STATS_PATH, "w") as f:
    f.writelines(stats_lines)
print(f"Saved {STATS_PATH}")

print("\nSUMMARY TABLE:")
print(pd.DataFrame(table_rows).to_string(index=False))

# =====================================================================
# BOX PLOTS  (2 rows x 3 cols)
# =====================================================================
COLORS = {"Normal": "#4A90D9", "Papilledema": "#E05C5C"}
feat_list = list(feature_labels.items())

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Optic Disc Features: Normal vs Papilledema\n(Bangladesh External Dataset, n=127 each)",
             fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes.flat, feat_list):
    n_vals = normal[col].dropna().values
    p_vals = papil[col].dropna().values

    bp = ax.boxplot([n_vals, p_vals], patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    bp["boxes"][0].set_facecolor(COLORS["Normal"])
    bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_facecolor(COLORS["Papilledema"])
    bp["boxes"][1].set_alpha(0.85)

    rng = np.random.default_rng(42)
    for i, vals in enumerate([n_vals, p_vals], 1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   alpha=0.35, s=12, color=list(COLORS.values())[i-1], zorder=3)

    _, pvalue = mannwhitneyu(n_vals, p_vals, alternative="two-sided")
    sig = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
    y_max = max(np.nanmax(n_vals), np.nanmax(p_vals))
    y_rng = y_max - min(np.nanmin(n_vals), np.nanmin(p_vals))
    pad   = y_rng * 0.07
    ax.plot([1,1,2,2], [y_max+pad, y_max+pad*1.5, y_max+pad*1.5, y_max+pad],
            color="black", linewidth=1)
    ax.text(1.5, y_max+pad*1.7, sig, ha="center", va="bottom", fontsize=11)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Normal", "Papilledema"], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)

patches = [mpatches.Patch(color=COLORS["Normal"], label="Normal"),
           mpatches.Patch(color=COLORS["Papilledema"], label="Papilledema")]
fig.legend(handles=patches, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {PLOT_PATH}")
plt.show()

print(f"\nCheck disc_debug/ folder — verify the green circles are on the optic disc")
print("If they are off, report back and the DISC_RADIUS_FRAC can be adjusted.")