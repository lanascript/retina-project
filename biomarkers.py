import glob
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# ================================
# CONFIG
# ================================
VESSEL_MASK_DIR = "test_drive/binary_vessels"
VISUALIZE_FIRST = True   # set False once you trust the pipeline

# ================================
# LOAD VESSEL MASKS
# ================================
vessel_files = sorted(glob.glob(f"{VESSEL_MASK_DIR}/*.png"))

if len(vessel_files) == 0:
    raise RuntimeError("No vessel masks found. Check path.")

print(f"Found {len(vessel_files)} vessel masks")

# ================================
# BIOMARKER 1: VESSEL DENSITY
# ================================
def vessel_density(mask):
    # approximate retinal FOV by excluding large black background
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.nan

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    roi = mask[min_y:max_y, min_x:max_x]
    return np.sum(roi) / roi.size


# ================================
# BIOMARKER 2: TORTUOSITY
# ================================
from skimage.morphology import thin
from scipy.ndimage import convolve

def mean_tortuosity(skeleton):
    # detect branch points
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])

    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
    branch_points = (neighbor_count >= 13)  # >2 neighbors

    # remove branch points to split into segments
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


# ================================
# BIOMARKER 3: FRACTAL DIMENSION
# ================================
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


# ================================
# MAIN LOOP (FIXED)
# ================================
densities = []
tortuosities = []
fractals = []

for idx, f in enumerate(vessel_files):

    img = imageio.imread(f)

    # ---- CORRECT handling for RGBA PNGs saved by matplotlib ----
    # RGB channels contain vessel data, alpha is always 255 (fully opaque)
    if img.ndim == 3 and img.shape[2] == 4:
        img_gray = np.mean(img[..., :3], axis=2)   # use RGB, ignore alpha
    elif img.ndim == 3:
        img_gray = img[..., 0]
    else:
        img_gray = img

    img_gray = img_gray.astype(np.float32)

    # Normalize if uint8
    if img_gray.max() > 1.0:
        img_gray = img_gray / 255.0

    # Final binary mask
    mask = img_gray > 0.5

    # Skeletonize
    skeleton = skeletonize(mask)

    # Compute biomarkers
    d = vessel_density(mask)
    t = mean_tortuosity(skeleton)
    fd = fractal_dimension(skeleton)

    densities.append(d)
    tortuosities.append(t)
    fractals.append(fd)

    # ---- VISUALIZE ONLY FIRST IMAGE ----
    if idx == 0 and VISUALIZE_FIRST:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Binary Vessel Mask")
        plt.imshow(mask.astype(np.float32), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Skeleton")
        plt.imshow(skeleton.astype(np.float32), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(mask, cmap="gray")
        plt.imshow(skeleton.astype(np.float32), cmap="hot", alpha=0.6)
        plt.axis("off")

        plt.show()


# ================================
# SUMMARY STATISTICS
# ================================
print("\n=== BIOMARKER SUMMARY (DRIVE) ===")
print(f"Vessel density  : mean={np.mean(densities):.4f}, std={np.std(densities):.4f}")
print(f"Tortuosity     : mean={np.nanmean(tortuosities):.4f}, std={np.nanstd(tortuosities):.4f}")
print(f"Fractal dim.   : mean={np.mean(fractals):.4f}, std={np.std(fractals):.4f}")

# Optional: save results
np.savez(
    "test_drive/vascular_biomarkers.npz",
    vessel_density=np.array(densities),
    tortuosity=np.array(tortuosities),
    fractal_dimension=np.array(fractals),
)

print("\nSaved biomarkers to test_drive/vascular_biomarkers.npz")
