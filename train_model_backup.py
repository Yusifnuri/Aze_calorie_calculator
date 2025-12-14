#!/usr/bin/env python3
"""
Optimized training script for imbalanced Azerbaijani food dataset
- Class weights to handle imbalance
- Data augmentation for minority classes
- Learning rate scheduling
- Early stopping
- Best model checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

# Device configuration
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"🖥️  Using device: {DEVICE}")


def compute_class_weights(dataset):
    """
    Compute class weights for imbalanced dataset
    Gives higher weight to minority classes
    """
    print("\n⚖️  Computing class weights...")
    
    # Extract all labels
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    labels = np.array(labels)  # Convert to numpy array
    unique_classes = np.unique(labels)
    
    # Compute balanced weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Show weights for minority classes
    weights_tensor = torch.FloatTensor(weights)
    print("\n   Top 5 classes with highest weights (minority classes):")
    sorted_indices = torch.argsort(weights_tensor, descending=True)
    for i in sorted_indices[:5]:
        class_name = dataset.classes[i]
        weight = weights_tensor[i].item()
        count = int(np.sum(labels == i))  # Fix: convert to int explicitly
        print(f"   - {class_name:30s} weight: {weight:.2f} ({count} images)")
    
    return weights_tensor


def get_data_transforms(img_size=224):
    """
    Data augmentation transforms
    More aggressive augmentation for training
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Good for food from different angles
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Random erasing to prevent overfitting
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform


def build_model(num_classes, freeze_layers=True):
    """
    Build EfficientNet-B0 model with pretrained weights
    """
    print("\n🏗️  Building model...")
    
    # Load pretrained model
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    
    # Freeze early layers for faster training and better generalization
    if freeze_layers:
        print("   Freezing early layers...")
        for param in list(model.parameters())[:-30]:  # Keep last 30 layers trainable
            param.requires_grad = False
    
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),  # Dropout to prevent overfitting
        nn.Linear(in_features, num_classes)
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   ✅ Model ready")
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    data_dir="data/train",
    epochs=40,
    batch_size=16,
    learning_rate=0.001,
    val_split=0.2,
    save_path="azeri_food_model.pt",
    early_stopping_patience=7
):
    """
    Main training function with class weights and early stopping
    """
    
    print("=" * 80)
    print("🚀 TRAINING AZERBAIJANI FOOD CLASSIFIER")
    print("=" * 80)
    print(f"📁 Data directory: {data_dir}")
    print(f"🔢 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"💾 Save path: {save_path}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load full dataset with training transforms
    print("\n📂 Loading dataset...")
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    print(f"   ✅ Loaded {len(full_dataset)} images")
    print(f"   ✅ Found {len(full_dataset.classes)} classes")
    
    # Show class distribution
    labels = [label for _, label in full_dataset]
    label_counts = Counter(labels)
    print("\n   Class distribution:")
    for class_idx, count in sorted(label_counts.items(), key=lambda x: x[1])[:5]:
        print(f"   - {full_dataset.classes[class_idx]:30s} {count:3d} images")
    print("   ...")
    
    # Compute class weights
    class_weights = compute_class_weights(full_dataset).to(DEVICE)
    
    # Split dataset
    print(f"\n🔀 Splitting dataset ({int((1-val_split)*100)}% train, {int(val_split*100)}% val)...")
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )
    
    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    
    # Build model
    model = build_model(len(full_dataset.classes), freeze_layers=True)
    model = model.to(DEVICE)
    
    # Loss function with class weights (CRITICAL for imbalanced data!)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\n📊 Results:")
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': full_dataset.classes,
                'class_weights': class_weights.cpu(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history,
            }
            
            torch.save(checkpoint, save_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"📁 Best model saved to: {save_path}")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Azerbaijani food classifier with class weights"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="azeri_food_model.pt",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience"
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        save_path=args.save,
        early_stopping_patience=args.patience
    )