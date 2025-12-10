# train.py
#
# Fine-tunes an EfficientNet-B0 model on the Azerbaijani food dataset.
# Expects the following structure:
#
#   project_root/
#     data/
#       train/
#         dolma/
#         plov/
#         ...
#       val/
#         dolma/
#         plov/
#         ...
#
# Saves the best model checkpoint as:
#   project_root/azeri_food_model.pt

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Select device (MPS for Apple Silicon, otherwise CUDA, then CPU)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_PATH = ROOT_DIR / "azeri_food_model.pt"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4


def get_dataloaders(data_dir: Path, batch_size: int):
    """
    Builds DataLoaders for the train and validation splits using ImageFolder.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names


def build_model(num_classes: int):
    """
    Loads EfficientNet-B0 pretrained on ImageNet and replaces the final
    classifier layer with a new Linear layer for our num_classes.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, epoch: int):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0

    print(f"Train Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, epoch: int):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0

    print(f"Val   Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def main():
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Num classes:", num_classes)

    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        _, val_acc = evaluate(model, val_loader, criterion, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                CHECKPOINT_PATH,
            )
            print(f"✅ Best model saved (val_acc={val_acc:.4f}) → {CHECKPOINT_PATH}")

    print("Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
