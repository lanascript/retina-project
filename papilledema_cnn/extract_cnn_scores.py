

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image

# =====================================================================
# CONFIG — adjust if paths differ
# =====================================================================
MODEL_PATH   = "models/best_model.pth"
NORMAL_DIR   = "data/bangladesh_external/Normal"
PAPIL_DIR    = "data/bangladesh_external/Papilledema"
OUTPUT_CSV   = "cnn_scores.csv"

# Must match your training setup
CLASS_NAMES  = ["Normal", "Papilledema", "Pseudopapilledema"]
NUM_CLASSES  = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

# =====================================================================
# LOAD MODEL
# =====================================================================
print("Loading EfficientNet-B0...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("  Model ready.\n")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
   
])

# =====================================================================
# INFERENCE
# =====================================================================
results = []

for class_label, folder in [("Normal", NORMAL_DIR), ("Papilledema", PAPIL_DIR)]:
    if not os.path.isdir(folder):
        print(f"WARNING: {folder} not found — skipping")
        continue

    files = [f for f in sorted(os.listdir(folder))
             if os.path.splitext(f)[1].lower() in VALID_EXTS]

    print(f"{class_label}: {len(files)} images")

    for fname in files:
        img_path = os.path.join(folder, fname)
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

            pred_idx   = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]

            results.append({
                "image":               fname,
                "class":               class_label,
                "cnn_prob_normal":     float(probs[0]),
                "cnn_prob_papil":      float(probs[1]),
                "cnn_prob_pseudo":     float(probs[2]),
                "cnn_pred":            pred_label,
                "cnn_correct":         int(pred_label == class_label or
                                          (class_label == "Papilledema" and pred_label == "Papilledema")),
            })

        except Exception as e:
            print(f"  ERROR {fname}: {e}")
            results.append({
                "image": fname, "class": class_label,
                "cnn_prob_normal": np.nan, "cnn_prob_papil": np.nan,
                "cnn_prob_pseudo": np.nan, "cnn_pred": "ERROR",
                "cnn_correct": 0,
            })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

# Summary
correct = df["cnn_correct"].sum()
total   = len(df)
print(f"\nCNN accuracy on Bangladesh: {correct}/{total} = {correct/total:.2%}")
print(f"Saved: {OUTPUT_CSV}")
print(f"\nNow copy {OUTPUT_CSV} into your Retinal-Vessel-Segmentation folder.")