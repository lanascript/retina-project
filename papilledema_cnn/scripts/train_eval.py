import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# TRANSFORMS (FIXED)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # ✅ match preprocessed images

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # reduced from 15

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # ✅ consistent
    transforms.ToTensor(),
])

# -----------------------------
# DATASETS
# -----------------------------
train_dataset = ImageFolder("data/kaggle_split/train_processed", transform=train_transform)
val_dataset = ImageFolder("data/kaggle_split/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 🔥 IMPORTANT: print class mapping
print("Class mapping:", train_dataset.classes)

# -----------------------------
# MODEL
# -----------------------------
model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

# 🔁 KEEP THIS SAME AS TRAINING
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

model = model.to(device)

# -----------------------------
# TRAINING SETUP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(10):

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss)
    print("Val Accuracy:", val_acc)
    print("Val F1:", val_f1)

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Best model saved!")

print("Training complete.")