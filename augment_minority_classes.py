#!/usr/bin/env python3
"""
Aggressive data augmentation for minority classes
Generates 10x more images from existing ones
"""

import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import json

# Classes that need augmentation (< 65 images)
MINORITY_CLASSES = [
    'baki_qurabiyesi', 'dolma', 'dovga', 'dushbere', 
    'lule_kabab', 'paxlava', 'plov', 'qutab',
    'seki_halvasi', 'shashlik', 'xash', 'yarpaq_xengel'
]

TARGET_IMAGES_PER_CLASS = 150  # Target after augmentation


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
    existing_images = list(class_path.glob("*.jpg")) + \
                     list(class_path.glob("*.jpeg")) + \
                     list(class_path.glob("*.png"))
    
    if not existing_images:
        print(f"  ⚠️  No images found in {class_name}")
        return 0
    
    current_count = len(existing_images)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"  ✅ {class_name}: Already has {current_count} images (target: {target_count})")
        return 0
    
    print(f"  🔄 {class_name}: {current_count} → {target_count} ({needed} new images)")
    
    # Create augmented directory
    aug_dir = class_path / "augmented"
    aug_dir.mkdir(exist_ok=True)
    
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
            aug_name = f"aug_{source_img_path.stem}_{'_'.join(aug_list[:3])}_{attempts}.jpg"
            save_path = aug_dir / aug_name
            
            aug_img.save(save_path, quality=95)
            generated += 1
            
        except Exception as e:
            print(f"    ⚠️  Error augmenting {source_img_path.name}: {e}")
        
        attempts += 1
    
    print(f"    ✅ Generated {generated} augmented images")
    return generated


def augment_dataset(data_dir="data/train", target_count=TARGET_IMAGES_PER_CLASS):
    """
    Augment all minority classes in the dataset
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    print("=" * 80)
    print("🔄 DATA AUGMENTATION FOR MINORITY CLASSES")
    print("=" * 80)
    print(f"\n📁 Dataset: {data_path}")
    print(f"🎯 Target: {target_count} images per class")
    print(f"🏷️  Classes to augment: {len(MINORITY_CLASSES)}")
    print()
    
    total_generated = 0
    
    for class_name in MINORITY_CLASSES:
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
    print("1. Move augmented images to main class folders:")
    print("   cd data/train/dolma")
    print("   mv augmented/* .")
    print()
    print("2. Run dataset analysis again:")
    print("   python data_analysis.py")
    print()
    print("3. Retrain model with balanced dataset:")
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
        default=MINORITY_CLASSES,
        help="Specific classes to augment"
    )
    
    args = parser.parse_args()
    
    if args.classes != MINORITY_CLASSES:
        MINORITY_CLASSES.clear()
        MINORITY_CLASSES.extend(args.classes)
    
    augment_dataset(args.data_dir, args.target)