import os
import shutil
import random

source_dir = "data/bangladesh_raw"
target_dir = "data/bangladesh_external"

os.makedirs(os.path.join(target_dir, "Normal"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "Papilledema"), exist_ok=True)

# Copy all Disc Edema (127)
disc_images = os.listdir(os.path.join(source_dir, "Discedema"))
for img in disc_images:
    shutil.copy(
        os.path.join(source_dir, "Discedema", img),
        os.path.join(target_dir, "Papilledema", img)
    )

# Randomly select 127 Healthy
healthy_images = os.listdir(os.path.join(source_dir, "Healthy"))
healthy_sample = random.sample(healthy_images, 127)

for img in healthy_sample:
    shutil.copy(
        os.path.join(source_dir, "Healthy", img),
        os.path.join(target_dir, "Normal", img)
    )

print("Bangladesh external dataset prepared.")