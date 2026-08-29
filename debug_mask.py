import glob
import imageio.v2 as imageio
import numpy as np
from skimage.morphology import skeletonize

vessel_files = sorted(glob.glob("test_drive/binary_vessels/*.png"))
if len(vessel_files) == 0:
    print("No mask files found in test_drive/binary_vessels/")
    exit(1)

f = vessel_files[0]
print(f"Loading: {f}")
img = imageio.imread(f)

print(f"Shape: {img.shape}, dtype: {img.dtype}")

# Check each channel
print("\nChannel analysis:")
for ch in range(img.shape[2]):
    vals = img[..., ch]
    print(f"  Channel {ch}: min={vals.min()}, max={vals.max()}, mean={vals.mean():.2f}, unique values: {len(np.unique(vals))}")

# Test with alpha channel (current approach)
mask_alpha = img[..., 3] > 127
skeleton_alpha = skeletonize(mask_alpha)
print(f"\nUsing Alpha (channel 3):")
print(f"  Mask % white: {100 * np.mean(mask_alpha):.2f}%")
print(f"  Skeleton % white: {100 * np.mean(skeleton_alpha):.2f}%")

# Test with RGB red channel (vessels often red)
mask_r = img[..., 0] > 127
skeleton_r = skeletonize(mask_r)
print(f"\nUsing Red channel only:")
print(f"  Mask % white: {100 * np.mean(mask_r):.2f}%")
print(f"  Skeleton % white: {100 * np.mean(skeleton_r):.2f}%")

# Test with inverted alpha
mask_alpha_inv = 255 - img[..., 3] > 127
skeleton_alpha_inv = skeletonize(mask_alpha_inv)
print(f"\nUsing Inverted Alpha:")
print(f"  Mask % white: {100 * np.mean(mask_alpha_inv):.2f}%")
print(f"  Skeleton % white: {100 * np.mean(skeleton_alpha_inv):.2f}%")
