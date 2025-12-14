# predict.py
#
# Predicts Azerbaijani dishes from an image with MULTI-DETECTION support
# Returns multiple dishes if confidence is above threshold

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import random
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from calories import NUTRITION_TABLE

# Thresholds - increased to reduce false positives
PRIMARY_CONFIDENCE_THRESHOLD = 0.40  # Increased from 0.30 to 0.40
SECONDARY_CONFIDENCE_THRESHOLD = 0.15  # Allow easier secondary detections
MAX_DETECTIONS = 5  # Maximum number of dishes to detect
OOD_MIN_TOP1 = 0.60          # treat as unknown if average top-1 confidence is too low
OOD_MIN_AGREEMENT = 0.55     # percentage of views that must agree on the same top-1 dish
OOD_MAX_STD_TOP1 = 0.22      # treat as unknown if top-1 confidence varies too much across views
UNKNOWN_LABEL = "unknown"
NON_MEAL_LABEL = "non_meal"
NON_MEAL_CONFIDENCE_THRESHOLD = 0.45
NON_MEAL_AUTO_THRESHOLD = 0.50  # below this confidence treat as non-food

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


def set_deterministic(seed: int = 42):
    """
    Ensure deterministic inference by fixing RNG state across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        # MPS does not currently expose additional determinism flags,
        # so we rely on manual seeds alone.
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some ops (e.g., on MPS) do not support strict determinism.
        pass


set_deterministic()


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


def get_infer_transform(image_size: int = 224):
    """
    Deterministic preprocessing for inference (no random augmentations).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(image: Union[str, Path, Image.Image]):
    """
    Deterministic preprocessing pipeline returning a single tensor.
    """
    infer_transform = get_infer_transform()

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    processed = infer_transform(img).unsqueeze(0)
    return processed


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
    model_bundle: Optional[Tuple[torch.nn.Module, List[str]]] = None,
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

    if model_bundle is not None:
        model, class_names = model_bundle
    else:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        model, class_names = load_model(checkpoint_path)

    input_tensor = preprocess_image(image)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        sorted_probs, sorted_indices = apply_visual_similarity_boost(
            sorted_probs, sorted_indices, class_names
        )

        primary_idx = int(sorted_indices[0])
        primary_dish = class_names[primary_idx]
        primary_confidence = float(sorted_probs[0])
        non_meal_detected = (
            primary_dish == NON_MEAL_LABEL
            and primary_confidence >= NON_MEAL_CONFIDENCE_THRESHOLD
        )
        non_meal_from_low_conf = primary_confidence < NON_MEAL_AUTO_THRESHOLD
        secondary_dish = None
        secondary_confidence = 0.0
        if len(sorted_probs) > 1:
            secondary_idx = int(sorted_indices[1])
            secondary_dish = class_names[secondary_idx]
            secondary_confidence = float(sorted_probs[1])

        is_confident = primary_confidence >= PRIMARY_CONFIDENCE_THRESHOLD
        if non_meal_detected or non_meal_from_low_conf:
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
                "primary_confidence": primary_confidence,
                "secondary_dish": secondary_dish,
                "secondary_confidence": secondary_confidence,
                "all_candidates": all_candidates,
                "all_classes": class_names,
                "is_confident": False,
                "non_meal_detected": True,
            }

        detected_dishes = []
        all_candidates = []

        for i in range(min(max_detections, len(sorted_probs))):
            idx = int(sorted_indices[i])
            dish_name = class_names[idx]
            confidence = float(sorted_probs[i])

            if dish_name == NON_MEAL_LABEL:
                continue

            threshold = PRIMARY_CONFIDENCE_THRESHOLD if i == 0 else SECONDARY_CONFIDENCE_THRESHOLD

            if confidence >= threshold:
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

            if confidence >= SECONDARY_CONFIDENCE_THRESHOLD * 0.5:
                all_candidates.append({
                    "dish": dish_name,
                    "confidence": confidence
                })

    if multi_detection and len(detected_dishes) > 1:
        detected_dishes = apply_post_processing_filter(detected_dishes, primary_confidence)

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
        "non_meal_detected": False,
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
