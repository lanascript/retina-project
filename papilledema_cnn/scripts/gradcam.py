# gradcam_visualization.py
# For use with Bangladesh external dataset (Normal / Papilledema)
# Model was trained on 3 classes: Normal, Papilledema, Pseudopapilledema

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH    = "models/best_model.pth"
IMAGE_FOLDER  = "test_images"       # put your Bangladesh images here
OUTPUT_FOLDER = "gradcam_outputs"

# Class index mapping from your training:
# 0 = Normal, 1 = Papilledema, 2 = Pseudopapilledema
CLASS_NAMES = ["Normal", "Papilledema", "Pseudopapilledema"]
PAPILLEDEMA_IDX = 1   # index of Papilledema class

# Set to True  → heatmap always shows Papilledema activation
# Set to False → heatmap shows whatever the model predicted
FORCE_PAPILLEDEMA_CAM = False

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 3)   # 3 classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# GRAD-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, input, output):
        self.activations = output

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        output[0, class_idx].backward()

        grads   = self.gradients[0]
        acts    = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

        return cam, class_idx, probs


grad_cam = GradCAM(model, model.features[-1])

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# -----------------------------
# OVERLAY
# -----------------------------
def overlay_cam(pil_image, cam):
    img_np      = np.array(pil_image.resize((224, 224)), dtype=np.float32) / 255.0
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay     = (heatmap_rgb + img_np)
    overlay     = overlay / overlay.max()
    overlay     = np.uint8(255 * overlay)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# -----------------------------
# PROCESS
# -----------------------------
image_paths = [
    os.path.join(IMAGE_FOLDER, f)
    for f in sorted(os.listdir(IMAGE_FOLDER))
    if os.path.splitext(f)[1].lower() in VALID_EXTS
]

if not image_paths:
    raise RuntimeError(f"No images found in '{IMAGE_FOLDER}'.")

print(f"Found {len(image_paths)} images\n")
print(f"{'File':<20} {'Predicted':<20} {'Normal%':>8} {'Papill%':>8} {'Pseudo%':>8}  CAM for")
print("-" * 80)

for img_path in image_paths:
    try:
        image  = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Which class to generate the heatmap FOR:
        # - If FORCE_PAPILLEDEMA_CAM=True:  always show papilledema activation
        # - If False: show activation for whatever the model predicted
        cam_class = PAPILLEDEMA_IDX if FORCE_PAPILLEDEMA_CAM else None

        cam, pred_class, probs = grad_cam.generate(tensor, class_idx=cam_class)

        pred_label = CLASS_NAMES[pred_class]
        cam_label  = CLASS_NAMES[PAPILLEDEMA_IDX if FORCE_PAPILLEDEMA_CAM else pred_class]
        filename   = os.path.basename(img_path)

        print(f"{filename:<20} {pred_label:<20} "
              f"{probs[0]*100:>7.1f}% {probs[1]*100:>7.1f}% {probs[2]*100:>7.1f}%  "
              f"[{cam_label}]")

        result    = overlay_cam(image, cam)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, result)

    except Exception as e:
        print(f"  ERROR {os.path.basename(img_path)}: {e}")

print(f"\nGrad-CAM images saved to: {OUTPUT_FOLDER}/")
print("\nWhat to check in the outputs:")
print("  GOOD → red/yellow heatmap centred on optic disc")
print("  BAD  → heatmap on image borders, background, or camera edges")