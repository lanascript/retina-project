import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.CenterCrop(350),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.CenterCrop(350),
    transforms.ToTensor(),
])

dataset = ImageFolder("data/bangladesh_external", transform=transform)
loader = DataLoader(dataset, batch_size=16)

model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)

model.load_state_dict(torch.load("models/best_model.pth"))
model = model.to(device)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:

        images = images.to(device)
        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
print(dataset.class_to_idx)