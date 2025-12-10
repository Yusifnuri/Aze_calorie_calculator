# predict.py
#
# Predicts the Azerbaijani dish from an image and returns
# full nutritional information (calories, fat, carbs, protein)
# based on NUTRITION_TABLE.
#
# Extended features:
# - returns top-k candidate dishes (for multiple-food / correction UI)
# - returns full class list stored in the checkpoint

from pathlib import Path
from typing import Union

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from calories import NUTRITION_TABLE

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT_DIR / "azeri_food_model.pt"


def build_model(num_classes: int):
    """
    Build the same EfficientNet-B0 architecture used in training,
    but without loading ImageNet weights (we load fine-tuned weights).
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path):
    """
    Load fine-tuned model checkpoint and class names.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def preprocess_image(image: Union[str, Path, Image.Image]):
    """
    Preprocess an image for EfficientNet-B0:
    - Resize to 224x224
    - Convert to tensor
    - Normalize with ImageNet statistics
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    tensor = transform(img).unsqueeze(0)
    return tensor


def predict_dish_and_nutrition(
    image: Union[str, Path, Image.Image],
    checkpoint_path: Union[str, Path, None] = None,
    top_k: int = 3,
):
    """
    Main prediction function.

    Parameters
    ----------
    image
        Path to image file or PIL.Image.
    checkpoint_path
        Optional override for model checkpoint path.
    top_k
        Number of top candidate dishes to return.

    Returns
    -------
    dict with keys:
        - dish: str (top-1 prediction)
        - confidence: float
        - nutrition: dict with calories, fat, carbs, protein (numeric)
        - top_candidates: list of {dish, confidence}
        - all_classes: list of str (all dish classes known by the model)
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model, class_names = load_model(checkpoint_path)

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # top-k candidates for extended features (multiple-food / correction)
        k = min(top_k, len(class_names))
        top_values, top_indices = torch.topk(probs, k=k)
        top_candidates = []
        for score, idx in zip(top_values, top_indices):
            idx = int(idx)
            top_candidates.append(
                {
                    "dish": class_names[idx],
                    "confidence": float(score),
                }
            )

    if pred_class not in NUTRITION_TABLE:
        raise KeyError(f"No nutrition info available for predicted dish: {pred_class}")

    nutrition = NUTRITION_TABLE[pred_class]

    return {
        "dish": pred_class,
        "confidence": confidence,
        "nutrition": nutrition,
        "top_candidates": top_candidates,
        "all_classes": class_names,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict Azerbaijani dish and nutrition from an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the food image.")

    args = parser.parse_args()
    image_path = args.image_path

    result = predict_dish_and_nutrition(image_path)

    dish = result["dish"]
    conf = result["confidence"]
    nut = result["nutrition"]

    print("Image:", image_path)
    print(f"Predicted dish: {dish}")
    print(f"Confidence: {conf*100:.2f}%")
    print("Nutrition (per portion):")
    print(f"  Calories: {nut['calories']} kcal")
    print(f"  Fat:      {nut['fat']} g")
    print(f"  Carbs:    {nut['carbs']} g")
    print(f"  Protein:  {nut['protein']} g")
