from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
CHECKPOINT_PATH = ROOT_DIR / "azeri_food_model.pt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# -------------------------------------------------------------------
# Dataloader
# -------------------------------------------------------------------
def get_dataloaders(batch_size: int = 32):

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Main dataset
    train_dataset_main = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    # User feedback dataset
    feedback_dir = DATA_DIR / "user_feedback"
    train_dataset_fb = None

    if feedback_dir.exists():
        subdirs = [p for p in feedback_dir.iterdir() if p.is_dir()]
        if subdirs:
            train_dataset_fb = datasets.ImageFolder(feedback_dir, transform=train_transform)
            print("User feedback samples:", len(train_dataset_fb))
        else:
            print("User feedback directory exists but has no class folders yet.")
    else:
        print("User feedback directory does not exist.")

    # Merge datasets
    if train_dataset_fb is not None:
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_main, train_dataset_fb]
        )
    else:
        train_dataset = train_dataset_main

    # Validation dataset
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # class names from ORIGINAL train folder ONLY
    class_names = train_dataset_main.classes

    return train_loader, val_loader, class_names


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
def build_model(num_classes: int):

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# -------------------------------------------------------------------
# Training functions
# -------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):

    model.train()
    running_loss = 0
    running_correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):

    model.eval()
    running_loss = 0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------
def main(num_epochs=10, lr=1e-4, batch_size=32):

    print("Using device:", DEVICE)

    train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)

    print("Number of classes:", len(class_names))
    print("Classes:", class_names)

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved new best model: val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
