# predict.py
#
# Predicts Azerbaijani dishes from an image with MULTI-DETECTION support
# Returns multiple dishes if confidence is above threshold

from collections import Counter
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from calories import NUTRITION_TABLE

# Thresholds - increased to reduce false positives
PRIMARY_CONFIDENCE_THRESHOLD = 0.40  # Increased from 0.30 to 0.40
SECONDARY_CONFIDENCE_THRESHOLD = 0.15  # Allow easier secondary detections
MAX_DETECTIONS = 5  # Maximum number of dishes to detect
OOD_MIN_TOP1 = 0.60          # avg top1 çox aşağıdırsa unknown
OOD_MIN_AGREEMENT = 0.55     # view-ların neçə faizi eyni top1 deyir
OOD_MAX_STD_TOP1 = 0.22      # top1 confidence view-lar üzrə çox oynayırsa unknown
UNKNOWN_LABEL = "unknown"

# Known confusion pairs - dishes that model often confuses
CONFUSION_PAIRS = {
    "ice_cream": ["dolma", "qutab", "pakhlava"],
    "fried_chicken": ["dolma", "qutab", "sarma"],
    "baki_qurabiyesi": ["dolma", "sarma", "qutab"],  # Added new confusion
    "dolma": ["ice_cream", "sarma", "fried_chicken", "baki_qurabiyesi"],
}

# Visual similarity rules - if primary is X and image has Y features, boost Y
VISUAL_SIMILARITY_RULES = {
    "fried_chicken": {
        "boost_if_green": ["dolma", "sarma", "qutab"],
        "boost_factor": 2.0,  # Increased from 1.5
    },
    "ice_cream": {
        "boost_if_wrapped": ["dolma", "sarma"],
        "boost_factor": 1.8,  # Increased from 1.4
    },
    "baki_qurabiyesi": {
        "boost_if_wrapped": ["dolma", "sarma", "qutab"],
        "boost_factor": 2.0,  # Strong boost for wrapped foods
    },
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT_DIR / "azeri_food_model.pt"


def build_model(num_classes: int):
    """Build EfficientNet-B0 architecture"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path):
    """
    Load fine-tuned model checkpoint and class names.
    Handles both old (Linear) and new (Sequential) architectures.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    
    # Check which architecture was used in training
    state_dict = checkpoint["model_state_dict"]
    
    # New training uses Sequential with Dropout
    if "classifier.1.1.weight" in state_dict:
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    # Old training uses simple Linear
    else:
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names

def preprocess_image(image: Union[str, Path, Image.Image], n_views: int = 8):
    """
    Create multiple augmented views of the same image for robustness / OOD gating.
    Returns: List[Tensor] each shape [1,3,224,224]
    """
    base = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    img = base(img)

    views = []
    for _ in range(n_views):
        views.append(aug(img).unsqueeze(0))  # [1,3,224,224]
    return views



def apply_visual_similarity_boost(sorted_probs, sorted_indices, class_names):
    """
    Apply visual similarity rules to boost likely confused classes
    """
    primary_idx = int(sorted_indices[0])
    primary_dish = class_names[primary_idx]
    
    # Check if primary dish has similarity rules
    if primary_dish not in VISUAL_SIMILARITY_RULES:
        return sorted_probs, sorted_indices
    
    rules = VISUAL_SIMILARITY_RULES[primary_dish]
    boost_candidates = rules.get("boost_if_green", []) + rules.get("boost_if_wrapped", [])
    boost_factor = rules.get("boost_factor", 1.3)
    
    # Create mutable copy of probabilities
    boosted_probs = sorted_probs.clone()
    
    # Boost probabilities of candidate dishes
    for i, idx in enumerate(sorted_indices):
        dish_name = class_names[int(idx)]
        if dish_name in boost_candidates and i > 0:  # Don't boost if already primary
            # Boost this probability
            boosted_probs[i] = boosted_probs[i] * boost_factor
    
    # Re-normalize probabilities
    boosted_probs = boosted_probs / boosted_probs.sum()
    
    # Re-sort after boosting
    sorted_boosted, sorted_boosted_indices = torch.sort(boosted_probs, descending=True)
    
    return sorted_boosted, sorted_boosted_indices


def apply_post_processing_filter(detected_dishes, top_confidence):
    """
    Post-processing to reduce false positives:
    - Remove dishes with very low confidence relative to top prediction
    - Apply confidence gap threshold
    - Check for known confusion pairs
    """
    if not detected_dishes:
        return detected_dishes
    
    filtered = []
    CONFIDENCE_GAP_THRESHOLD = 0.45  # Secondary dish must be at least 45% of primary by default
    
    primary_dish_name = detected_dishes[0]['dish'] if detected_dishes else None
    
    for i, dish_info in enumerate(detected_dishes):
        if i == 0:  # Always keep primary dish
            filtered.append(dish_info)
        else:
            dish_name = dish_info['dish']
            confidence_ratio = dish_info['confidence'] / top_confidence
            
            # Check if this is a known confusion pair
            is_confusion = False
            if primary_dish_name in CONFUSION_PAIRS:
                if dish_name in CONFUSION_PAIRS[primary_dish_name]:
                    is_confusion = True
            
            # Skip if it's a confusion pair AND confidence is low
            if is_confusion and confidence_ratio < 0.70:
                continue
            
            min_ratio = SIDE_DISH_RATIO_OVERRIDES.get(dish_name, CONFIDENCE_GAP_THRESHOLD)
            if confidence_ratio >= min_ratio:
                filtered.append(dish_info)
    
    return filtered


def predict_dish_and_nutrition(
    image: Union[str, Path, Image.Image],
    checkpoint_path: Union[str, Path, None] = None,
    multi_detection: bool = True,
    max_detections: int = MAX_DETECTIONS,
):
    """
    Main prediction function with MULTI-DETECTION support.

    Returns a dict:
        - detected_dishes: list of {dish, confidence, nutrition}
        - primary_dish: str | None (highest confidence dish)
        - all_candidates: list of all dishes above threshold
        - all_classes: list of all class names
        - is_confident: bool (whether primary dish is confident enough)
    """

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load model and class names
    model, class_names = load_model(checkpoint_path)

    # Preprocess image - returns multiple views
    input_tensors = preprocess_image(image)
    
    # Ensure all tensors are 4D [batch, channels, height, width]
    input_tensors = [t if t.dim() == 4 else t.unsqueeze(0) for t in input_tensors]
    
    # Ensemble prediction - average predictions from multiple views
    all_probs = []
    
    with torch.no_grad():
        for input_tensor in input_tensors:
            input_tensor = input_tensor.to(DEVICE)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs)
        
        # Average probabilities across all views
        avg_probs = torch.stack(all_probs).mean(dim=0)

        # ---- OOD / NOT-FOOD GATE (stability across views) ----
        view_top1 = [int(p.argmax().item()) for p in all_probs]
        view_top1_probs = [float(p.max().item()) for p in all_probs]

        most_common_idx, count = Counter(view_top1).most_common(1)[0]
        agreement = count / len(view_top1)

        avg_top1 = float(avg_probs.max().item())
        std_top1 = float(torch.tensor(view_top1_probs).std().item())

        is_ood = (
            (avg_top1 < OOD_MIN_TOP1)
            or ((agreement < OOD_MIN_AGREEMENT) and (std_top1 > OOD_MAX_STD_TOP1))
        )

        if is_ood:
            sorted_probs, sorted_indices = torch.sort(avg_probs, descending=True)
            secondary_dish = None
            secondary_confidence = 0.0
            if len(sorted_probs) > 1:
                secondary_idx = int(sorted_indices[1])
                secondary_dish = class_names[secondary_idx]
                secondary_confidence = float(sorted_probs[1])

            all_candidates = []
            for i in range(min(5, len(sorted_probs))):
                idx = int(sorted_indices[i])
                all_candidates.append({
                    "dish": class_names[idx],
                    "confidence": float(sorted_probs[i])
                })

            return {
                "detected_dishes": [],
                "primary_dish": None,
                "primary_confidence": avg_top1,
                "secondary_dish": secondary_dish,
                "secondary_confidence": secondary_confidence,
                "all_candidates": all_candidates,
                "all_classes": class_names,
                "is_confident": False,
                "ood_debug": {
                    "avg_top1": avg_top1,
                    "agreement": agreement,
                    "std_top1": std_top1
                }
            }
        # ---- end OOD gate ----
        
        # Get sorted predictions
        sorted_probs, sorted_indices = torch.sort(avg_probs, descending=True)
        
        # Apply visual similarity boost for known confusions
        sorted_probs, sorted_indices = apply_visual_similarity_boost(
            sorted_probs, sorted_indices, class_names
        )

        # Primary dish (highest confidence after boosting)
        primary_idx = int(sorted_indices[0])
        primary_dish = class_names[primary_idx]
        primary_confidence = float(sorted_probs[0])
        secondary_dish = None
        secondary_confidence = 0.0
        if len(sorted_probs) > 1:
            secondary_idx = int(sorted_indices[1])
            secondary_dish = class_names[secondary_idx]
            secondary_confidence = float(sorted_probs[1])

        # Check if primary dish is confident enough
        is_confident = primary_confidence >= PRIMARY_CONFIDENCE_THRESHOLD

        # Multi-detection: find all dishes above threshold
        detected_dishes = []
        all_candidates = []

        for i in range(min(max_detections, len(sorted_probs))):
            idx = int(sorted_indices[i])
            dish_name = class_names[idx]
            confidence = float(sorted_probs[i])

            # Determine threshold based on position
            threshold = PRIMARY_CONFIDENCE_THRESHOLD if i == 0 else SECONDARY_CONFIDENCE_THRESHOLD

            if confidence >= threshold:
                # Get nutrition info
                nutrition = NUTRITION_TABLE.get(dish_name, {
                    "calories": 0,
                    "fat": 0,
                    "carbs": 0,
                    "protein": 0
                })

                dish_info = {
                    "dish": dish_name,
                    "confidence": confidence,
                    "nutrition": nutrition,
                    "is_primary": (i == 0)
                }

                if multi_detection:
                    detected_dishes.append(dish_info)
                elif i == 0:
                    detected_dishes.append(dish_info)

            # Keep track of all reasonable candidates for UI selection
            if confidence >= SECONDARY_CONFIDENCE_THRESHOLD * 0.5:
                all_candidates.append({
                    "dish": dish_name,
                    "confidence": confidence
                })

    # Apply post-processing filter to reduce false positives
    if multi_detection and len(detected_dishes) > 1:
        detected_dishes = apply_post_processing_filter(detected_dishes, primary_confidence)

    # If not confident enough, return None for primary dish
    if not is_confident:
        primary_dish = None

    return {
        "detected_dishes": detected_dishes,
        "primary_dish": primary_dish,
        "primary_confidence": primary_confidence,
        "secondary_dish": secondary_dish,
        "secondary_confidence": secondary_confidence,
        "all_candidates": all_candidates,
        "all_classes": class_names,
        "is_confident": is_confident,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict Azerbaijani dishes and nutrition from an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the food image.")
    parser.add_argument("--single", action="store_true", help="Detect only single dish")

    args = parser.parse_args()
    image_path = args.image_path

    result = predict_dish_and_nutrition(
        image_path, 
        multi_detection=not args.single
    )

    detected = result["detected_dishes"]
    is_confident = result["is_confident"]

    if not is_confident:
        print("⚠️  Model is not confident enough in its prediction.")
        print(f"Primary guess confidence: {result['primary_confidence']*100:.2f}%")
        print("\nTop candidates:")
        for c in result["all_candidates"][:5]:
            print(f"  - {c['dish']}: {c['confidence']*100:.1f}%")
    else:
        print(f"✅ Detected {len(detected)} dish(es) in the image:\n")
        
        for i, d in enumerate(detected, 1):
            marker = "🍽️ " if d["is_primary"] else "  "
            print(f"{marker}{i}. {d['dish']}")
            print(f"   Confidence: {d['confidence']*100:.2f}%")
            print(f"   Nutrition (per portion):")
            print(f"     Calories: {d['nutrition']['calories']} kcal")
            print(f"     Fat: {d['nutrition']['fat']} g")
            print(f"     Carbs: {d['nutrition']['carbs']} g")
            print(f"     Protein: {d['nutrition']['protein']} g")
            print()
# Some dishes regularly appear as sides (e.g., fries with burgers)
SIDE_DISH_RATIO_OVERRIDES = {
    "french_fries": 0.35,
    "salad": 0.35,
}
