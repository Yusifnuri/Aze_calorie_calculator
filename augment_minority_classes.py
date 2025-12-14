#!/usr/bin/env python3
"""
Aggressive data augmentation for minority classes.
Automatically finds classes with fewer than TARGET images and augments them.
"""

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import uuid
from typing import Dict, List, Optional

TARGET_IMAGES_PER_CLASS = 150  # Target after augmentation
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def collect_image_files(class_path: Path):
    """Return all supported image files in a class directory."""
    files = []
    for pattern in IMAGE_PATTERNS:
        files.extend(class_path.glob(pattern))
    return files


def compute_class_counts(data_path: Path) -> Dict[str, int]:
    """Count images per class directory."""
    counts = {}
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(collect_image_files(class_dir))
    return counts


def detect_minority_classes(class_counts: Dict[str, int], target_count: int) -> List[str]:
    """Return classes that have fewer images than the target."""
    return sorted(
        [cls for cls, count in class_counts.items() if count < target_count],
        key=lambda cls: class_counts[cls]
    )


def augment_image(img, seed=None):
    """
    Apply random augmentation to an image
    """
    if seed is not None:
        random.seed(seed)
    
    augmentations = []
    
    # 1. Rotation (-30 to +30 degrees)
    if random.random() > 0.3:
        angle = random.randint(-30, 30)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
        augmentations.append(f"rot{angle}")
    
    # 2. Horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append("flip")
    
    # 3. Brightness adjustment
    if random.random() > 0.4:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
        augmentations.append(f"bright{factor:.2f}")
    
    # 4. Contrast adjustment
    if random.random() > 0.4:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
        augmentations.append(f"contrast{factor:.2f}")
    
    # 5. Color saturation
    if random.random() > 0.4:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
        augmentations.append(f"color{factor:.2f}")
    
    # 6. Slight blur (for realism)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        augmentations.append("blur")
    
    # 7. Random crop and resize
    if random.random() > 0.5:
        width, height = img.size
        crop_percent = random.uniform(0.8, 0.95)
        
        new_width = int(width * crop_percent)
        new_height = int(height * crop_percent)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        img = img.crop((left, top, left + new_width, top + new_height))
        img = img.resize((width, height), Image.LANCZOS)
        augmentations.append("crop")
    
    # 8. Sharpness
    if random.random() > 0.6:
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.8, 1.5)
        img = enhancer.enhance(factor)
        augmentations.append(f"sharp{factor:.2f}")
    
    return img, augmentations


def augment_class(class_dir, target_count=TARGET_IMAGES_PER_CLASS):
    """
    Augment a single class directory to reach target count
    """
    class_path = Path(class_dir)
    class_name = class_path.name
    
    # Get existing images
    existing_images = collect_image_files(class_path)
    
    if not existing_images:
        print(f"  ⚠️  No images found in {class_name}")
        return 0
    
    current_count = len(existing_images)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"  ✅ {class_name}: Already has {current_count} images (target: {target_count})")
        return 0
    
    print(f"  🔄 {class_name}: {current_count} → {target_count} ({needed} new images)")
    
    generated = 0
    attempts = 0
    max_attempts = needed * 3  # Safety limit
    
    while generated < needed and attempts < max_attempts:
        # Pick random source image
        source_img_path = random.choice(existing_images)
        
        try:
            # Load and augment
            img = Image.open(source_img_path).convert('RGB')
            aug_img, aug_list = augment_image(img)
            
            # Save with descriptive name
            aug_suffix = "_".join(aug_list[:3]) if aug_list else "base"
            aug_name = f"aug_{source_img_path.stem}_{aug_suffix}_{uuid.uuid4().hex[:6]}.jpg"
            save_path = class_path / aug_name
            
            aug_img.save(save_path, quality=95)
            generated += 1
            
        except Exception as e:
            print(f"    ⚠️  Error augmenting {source_img_path.name}: {e}")
        
        attempts += 1
    
    print(f"    ✅ Generated {generated} augmented images")
    return generated


def augment_dataset(
    data_dir="data/train",
    target_count=TARGET_IMAGES_PER_CLASS,
    specific_classes: Optional[List[str]] = None,
):
    """
    Augment all minority classes in the dataset
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    class_counts = compute_class_counts(data_path)
    if not class_counts:
        print(f"❌ No class folders found in {data_path}")
        return

    if specific_classes:
        classes_to_augment = []
        for class_name in specific_classes:
            class_dir = data_path / class_name
            if not class_dir.exists():
                print(f"  ⚠️  Class directory not found: {class_name}")
                continue
            classes_to_augment.append(class_name)
    else:
        classes_to_augment = detect_minority_classes(class_counts, target_count)
    
    if not classes_to_augment:
        print("=" * 80)
        print("✅ All classes already meet or exceed the target image count!")
        print("=" * 80)
        return

    print("=" * 80)
    print("🔄 DATA AUGMENTATION FOR MINORITY CLASSES")
    print("=" * 80)
    print(f"\n📁 Dataset: {data_path}")
    print(f"🎯 Target: {target_count} images per class")
    source_msg = "auto-detected (below target)" if not specific_classes else "manual selection"
    print(f"🏷️  Classes to augment: {len(classes_to_augment)} [{source_msg}]")
    for class_name in classes_to_augment:
        current = class_counts.get(class_name, 0)
        print(f"   - {class_name:25s} {current:4d} images")
    print()
    
    total_generated = 0
    
    for class_name in classes_to_augment:
        class_dir = data_path / class_name
        
        if not class_dir.exists():
            print(f"  ⚠️  Class directory not found: {class_name}")
            continue
        
        generated = augment_class(class_dir, target_count)
        total_generated += generated
        print()
    
    print("=" * 80)
    print(f"✅ AUGMENTATION COMPLETE")
    print(f"📊 Total new images generated: {total_generated}")
    print("=" * 80)
    print()
    print("📋 NEXT STEPS:")
    print("1. Run dataset analysis again:")
    print("   python data_analysis.py")
    print()
    print("2. Retrain model with balanced dataset:")
    print("   python train.py --use_class_weights")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment minority classes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_IMAGES_PER_CLASS,
        help="Target number of images per class"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional list of classes to augment (defaults to all classes below target)"
    )
    
    args = parser.parse_args()
    augment_dataset(args.data_dir, args.target, args.classes)
