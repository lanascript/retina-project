import os
import glob
import sys
import numpy as np
import imageio.v2 as imageio
import pandas as pd
from PIL import Image

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import convolve

# Add lib to path for helper functions
sys.path.insert(0, './lib/')
from help_functions import load_hdf5
from extract_patches import get_data_testing_overlap, recompone_overlap, kill_border
from pre_processing import my_PreProc
from nn_models_v2 import get_unet

# Keras
from keras.models import load_model


# =====================================
# CONFIG
# =====================================
DATA_ROOT = "."
CLASS_FOLDERS = {
    "0.0.Normal": "Normal",
    "10.0.Possible glaucoma": "Glaucoma",
    "10.1.Optic atrophy": "Atrophy"
}

MASK_ROOT = "processed_masks"
RESULTS_CSV = "retinal_biomarkers_full.csv"

# Path to trained model
MODEL_PATH = "./test_drive/test_drive_best_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")
# =====================================
# BIOMARKER FUNCTIONS (YOUR CODE)
# =====================================

def vessel_density(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.nan

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    roi = mask[min_y:max_y, min_x:max_x]
    return np.sum(roi) / roi.size


def mean_tortuosity(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])

    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
    branch_points = (neighbor_count >= 13)

    pruned = skeleton.copy()
    pruned[branch_points] = 0

    labeled = label(pruned)
    torts = []

    for region in regionprops(labeled):
        coords = region.coords
        if len(coords) < 5:
            continue

        path_len = np.sum(
            np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        )
        euclid = np.linalg.norm(coords[0] - coords[-1])

        if euclid > 0:
            torts.append(path_len / euclid)

    if len(torts) == 0:
        return np.nan

    return np.mean(torts)


def fractal_dimension(Z):
    Z = Z.astype(bool)

    ys, xs = np.where(Z)
    if len(xs) == 0:
        return np.nan

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    Z = Z[min_y:max_y, min_x:max_x]

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )
        return np.count_nonzero(S)

    p = min(Z.shape)
    n = int(2 ** np.floor(np.log2(p)))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

    counts = []
    for size in sizes:
        c = boxcount(Z, int(size))
        if c > 0:
            counts.append(c)

    if len(counts) == 0:
        return np.nan

    sizes = sizes[:len(counts)]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]


# =====================================
# SINGLE-IMAGE PREDICTION FUNCTION
# =====================================

def predict_single_image(image_path, model, patch_height=48, patch_width=48):
    """
    Predict vessel mask for a single image using trained U-Net model.
    Uses simple sliding window with no overlapping reconstruction.
    
    Args:
        image_path: Path to input retinal image
        model_path: Path to trained .keras model
        patch_height, patch_width: Patch dimensions (must match training)
    
    Returns:
        Binary vessel mask (0/1 uint8)
    """
    
    # Load model
    # model = load_model(model_path)
    
    # Load image and convert to grayscale
    from pre_processing import my_PreProc
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.uint8)

    # Convert to channels-first and add batch dim: (H,W,3) -> (1,3,H,W)
    img_cf = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]

    # Apply same preprocessing as training (expects shape (N,3,H,W))
    proc = my_PreProc(img_cf)

    # my_PreProc returns (N,1,H,W) -> extract original H,W
    original_shape = proc.shape[2], proc.shape[3]

    # Convert to (H, W, 1) for downstream code
    img_array = proc[0, 0, :, :][:, :, np.newaxis]
    
    h, w = original_shape
    
    # Pad image to multiple of patch size
    padded_h = int(np.ceil(h / patch_height) * patch_height)
    padded_w = int(np.ceil(w / patch_width) * patch_width)
    
    padded_img = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
    padded_img[:h, :w, :] = img_array
    
    # Extract patches in a grid
    patches = []
    coords = []
    
    for y in range(0, padded_h, patch_height):
        for x in range(0, padded_w, patch_width):
            patch = padded_img[y:y+patch_height, x:x+patch_width, :]
            assert patch.shape == (patch_height, patch_width, 1), f"Patch shape {patch.shape} != expected (48, 48, 1)"
            patches.append(patch)
            coords.append((y, x))
    
    # Stack into batch
    patches = np.array(patches)  # (N, 48, 48, 1)
    
    # Predict all patches at once
    print(f"    Predicting {len(patches)} patches...", end=" ", flush=True)
    predictions = model.predict(patches, verbose=0)
    
    # Reshape to spatial format if model outputs flattened predictions
    # Model outputs (N, 2304, 2) which needs to be (N, 48, 48, 2)
    if len(predictions.shape) == 3 and predictions.shape[1] == 2304:
        predictions = predictions.reshape((-1, patch_height, patch_width, 2))
    
    # Extract vessel channel (channel 1 = vessel)
    vessel_maps = predictions[:, :, :, 1]  # (N, 48, 48)
    
    # Reconstruct full image
    pred_full = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    for (y, x), vessel_map in zip(coords, vessel_maps):
        pred_full[y:y+patch_height, x:x+patch_width] = vessel_map
    
    # Crop back to original size
    pred_full = pred_full[:h, :w]
    
    # Apply threshold
    threshold = 0.4
    binary_mask = (pred_full >= threshold).astype(np.uint8)
    
    return binary_mask



# =====================================
# MAIN LOOP
# =====================================

results = []

# Path to trained model
MODEL_PATH = "./test_drive/test_drive_best_model.keras"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    exit(1)

for folder, label_name in CLASS_FOLDERS.items():

    image_paths = (
    glob.glob(os.path.join(DATA_ROOT, folder, "*.jpg")) +
    glob.glob(os.path.join(DATA_ROOT, folder, "*.JPG")) +
    glob.glob(os.path.join(DATA_ROOT, folder, "*.png")) +
    glob.glob(os.path.join(DATA_ROOT, folder, "*.jpeg"))
)

    print(f"\nProcessing {label_name} ({len(image_paths)} images)")

    os.makedirs(os.path.join(MASK_ROOT, label_name), exist_ok=True)

    for idx, img_path in enumerate(image_paths):

        filename = os.path.basename(img_path)
        print(f"  [{idx+1}/{len(image_paths)}] {filename}...", end=" ", flush=True)
        
        try:
            # Predict vessel mask directly
            mask = predict_single_image(img_path, model)
            
            # Save mask
            mask_path = os.path.join(MASK_ROOT, label_name, filename.rsplit('.', 1)[0] + '.png')
            imageio.imwrite(mask_path, mask * 255)
            
            # Compute biomarkers
            skeleton = skeletonize(mask > 0)
            
            d = vessel_density(mask)
            t = mean_tortuosity(skeleton)
            fd = fractal_dimension(skeleton)
            
            results.append({
                "image": filename,
                "class": label_name,
                "vessel_density": d,
                "tortuosity": t,
                "fractal_dimension": fd
            })
            
            print(f"✓ (density={d:.3f}, tort={t:.3f}, fd={fd:.3f})")
        
        except Exception as e:
            import traceback
            # Print exception type, message and full traceback for debugging
            print("✗ ERROR:", type(e).__name__, str(e))
            traceback.print_exc()
            results.append({
                "image": filename,
                "class": label_name,
                "vessel_density": np.nan,
                "tortuosity": np.nan,
                "fractal_dimension": np.nan
            })


# =====================================
# SAVE CSV
# =====================================

df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

print("\nSaved:", RESULTS_CSV)