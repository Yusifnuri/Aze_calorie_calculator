#!/usr/bin/env python3
"""
Analyze training dataset to identify class imbalance issues
"""

from pathlib import Path
from collections import Counter
import json

def analyze_dataset(data_dir="data/train"):
    """
    Analyze dataset structure and class distribution
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        print("Please provide correct path to your training data")
        return
    
    # Count images per class
    class_counts = {}
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            # Count images in this class
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            class_counts[class_dir.name] = len(image_files)
    
    if not class_counts:
        print("❌ No class directories found!")
        return
    
    # Calculate statistics
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)
    avg_per_class = total_images / num_classes
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print report
    print("=" * 80)
    print("📊 DATASET ANALYSIS REPORT")
    print("=" * 80)
    print(f"\n📁 Dataset path: {data_path}")
    print(f"📷 Total images: {total_images}")
    print(f"🏷️  Total classes: {num_classes}")
    print(f"📈 Average per class: {avg_per_class:.1f}")
    print(f"📊 Min images: {sorted_classes[-1][1]}")
    print(f"📊 Max images: {sorted_classes[0][1]}")
    
    # Identify problem classes
    min_threshold = avg_per_class * 0.5  # Less than 50% of average
    problem_classes = [c for c, count in class_counts.items() if count < min_threshold]
    
    print("\n" + "=" * 80)
    print("⚠️  CLASSES WITH INSUFFICIENT DATA (< 50% of average):")
    print("=" * 80)
    
    if problem_classes:
        for cls in sorted(problem_classes):
            count = class_counts[cls]
            percentage = (count / avg_per_class) * 100
            print(f"  ❌ {cls:30s} {count:4d} images ({percentage:.0f}% of avg)")
            print(f"      → Need at least {int(min_threshold - count)} more images")
    else:
        print("  ✅ All classes have sufficient data!")
    
    # Show top and bottom classes
    print("\n" + "=" * 80)
    print("📊 TOP 10 CLASSES (Most Images):")
    print("=" * 80)
    for cls, count in sorted_classes[:10]:
        bar = "█" * int(count / sorted_classes[0][1] * 50)
        print(f"  {cls:30s} {count:4d} {bar}")
    
    print("\n" + "=" * 80)
    print("⚠️  BOTTOM 10 CLASSES (Least Images):")
    print("=" * 80)
    for cls, count in sorted_classes[-10:]:
        bar = "█" * int(count / sorted_classes[0][1] * 50)
        status = "❌" if count < min_threshold else "⚠️"
        print(f"  {status} {cls:30s} {count:4d} {bar}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    if len(problem_classes) > 0:
        print(f"\n1. 📸 COLLECT MORE DATA for {len(problem_classes)} classes")
        print(f"   - Target: at least {int(avg_per_class)} images per class")
        print(f"   - Priority classes: {', '.join(problem_classes[:5])}")
        
        print(f"\n2. 🔄 DATA AUGMENTATION")
        print(f"   - Apply aggressive augmentation to underrepresented classes")
        print(f"   - Use rotation, flip, color jitter, crop variations")
        
        print(f"\n3. ⚖️ CLASS WEIGHTS during training")
        print(f"   - Use weighted loss to balance classes")
        print(f"   - Give higher weight to minority classes")
        
        print(f"\n4. 🔀 OVERSAMPLING minority classes")
        print(f"   - Duplicate images from small classes")
        print(f"   - Use SMOTE or similar techniques")
    else:
        print("  ✅ Dataset is well-balanced!")
    
    # Save report to file
    report_path = Path("dataset_analysis_report.json")
    report_data = {
        "total_images": total_images,
        "num_classes": num_classes,
        "avg_per_class": avg_per_class,
        "class_counts": class_counts,
        "problem_classes": problem_classes,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")
    print("=" * 80)
    
    return class_counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset class distribution")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Path to training data directory"
    )
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir)