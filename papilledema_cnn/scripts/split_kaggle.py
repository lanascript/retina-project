import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "data/kaggle_raw"
output_dir = "data/kaggle_split"

classes = ["Normal", "Papilledema", "Pseudopapilledema"]

for cls in classes:
    images = os.listdir(os.path.join(base_dir, cls))
    
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split, split_data in zip(["train", "val", "test"], [train, val, test]):
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
        for img in split_data:
            shutil.copy(
                os.path.join(base_dir, cls, img),
                os.path.join(output_dir, split, cls, img)
            )

print("Kaggle dataset split completed.")