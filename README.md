# Azerbaijani Cuisine Calorie Model

A computer vision application that recognizes Azerbaijani and selected international dishes, looks up portion-based nutrition estimates, and presents the results in a Streamlit interface.

**Stack:** Python · PyTorch · EfficientNet-B0 · Streamlit · OpenAI API

## What it does

- Fine-tunes an EfficientNet-B0 classifier with weighted loss for imbalanced food classes.
- Uses multi-view prediction and heuristics to propose dishes from an image.
- Looks up calories and macronutrients in a dish nutrition table.
- Supports manual dish corrections, portion adjustments, and combined nutrition totals.
- Generates personalized text explanations through the OpenAI API.

Nutrition values come from a lookup table and portion assumptions. They are not measurements of the photographed meal's weight or ingredients. Multi-dish predictions use heuristics rather than a separately evaluated object detection model.

## Recorded training result

The committed [training history](azeri_food_model_history.json) contains **19 epochs**, with a highest recorded validation accuracy of **97.07%** at epoch 13. This is a validation metric from the stored run, not independent test accuracy or evidence of calorie-estimation accuracy.

Before making generalization claims, evaluate on a separate set of original images and report per-class performance. If augmented copies are created before a random split, related images can appear on both sides of the split and inflate validation scores.

## Quick start

```bash
git clone https://github.com/Yusifnuri/Aze_calorie_calculator.git
cd Aze_calorie_calculator
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Choose a PyTorch installation compatible with your device. The predictor selects CUDA, MPS, or CPU according to availability.

### Run a local prediction

The default checkpoint is `azeri_food_model.pt` in the repository root.

```bash
python src/predict.py /path/to/food.jpg
# Disable multi-dish prediction:
python src/predict.py /path/to/food.jpg --single
```

The CLI does not accept a `--checkpoint` option. To use a different checkpoint programmatically, pass `checkpoint_path` to `predict_dish_and_nutrition` in `src/predict.py`.

### Launch the interface

Copy `.env.example` to `.env` and set your own `OPENAI_API_KEY`, then run:

```bash
python -m streamlit run app/ui.py
```

The UI imports the explanation module at startup, and that module initializes an OpenAI client immediately. A key is therefore currently required to initialize the UI even if you later disable explanations in the sidebar. The prediction CLI can be used without the explanation module.

Upload an image, review the predicted dishes, correct labels or portions where needed, and inspect the resulting nutrition totals. Enabling explanations makes API calls using your key.

## Dataset and training

The primary training script reads a single ImageFolder directory:

```text
data/train/
├── dish_class_a/
│   ├── image_01.jpg
│   └── image_02.jpg
├── dish_class_b/
└── non_meal/           Optional non-food class
```

It creates a seeded random train/validation split inside that directory using `--val_split`. It does **not** accept separate `--train_dir` and `--val_dir` arguments, or read `data/val` as an external validation set.

```bash
python data_analysis.py --data_dir data/train
python train_model.py   --data_dir data/train   --epochs 25   --batch_size 32   --val_split 0.2   --save azeri_food_model.pt
```

Training saves the best checkpoint and writes epoch losses and accuracies to the corresponding `_history.json` file. `--lr` and `--patience` control learning rate and early stopping.

`augment_minority_classes.py` can expand small classes, but preserve an untouched evaluation set before creating derivatives:

```bash
python augment_minority_classes.py --data_dir data/train --target 150
```

The current trainer shares one underlying ImageFolder between the two split subsets. Setting the validation transform also changes that shared dataset's training transform. Separate dataset instances/transforms and an original-image split should be implemented before treating this as a rigorous augmentation experiment.

## How the pieces fit together

```text
Food image → EfficientNet-B0 predictions → Dish/portion correction
                                               ↓
                                    Nutrition table lookup
                                               ↓
                                Calories and macronutrients
                                               ↓
                                 Optional AI explanation
```

| Path | Purpose |
| --- | --- |
| `app/ui.py` | Streamlit application |
| `src/predict.py` | Model loading and image prediction |
| `src/calories.py` | Dish nutrition table |
| `src/genai_explainer.py` | OpenAI text explanations |
| `train_model.py` | Main training entry point |
| `data_analysis.py` | Class counts and dataset analysis |
| `augment_minority_classes.py` | Minority-class image augmentation |
| `azeri_food_model_history.json` | Recorded epoch metrics |

## Troubleshooting

- **Checkpoint missing:** confirm `azeri_food_model.pt` is present in the repository root or train a new checkpoint.
- **Poor predictions:** inspect class coverage, image quality, confidence, and non-food examples. Manual corrections remain useful for unfamiliar dishes.
- **UI fails before loading:** check the OpenAI key, dependency installation, and error message; client initialization happens at import time.
- **Device errors:** install a compatible PyTorch build or use CPU for prediction.

## Development priorities

Separate train/validation transforms, reserve an independent test set before augmentation, add per-class metrics and a confusion matrix, and initialize the explanation client only when needed. These are improvement opportunities, not features already implemented.

## Contributions and reuse

Describe dataset provenance, class coverage, and evaluation changes in contributions. Review image rights and the terms of upstream models/dependencies before redistributing artifacts; this repository currently has no root license file granting reuse permissions.
