# Azerbaijani Cuisine Calorie Model

Image classification pipeline that recognises Azerbaijani and selected international dishes, estimates macronutrients per portion, and serves results through a Streamlit UI backed by OpenAI powered explanations.

## Key Features
- EfficientNet-B0 fine-tuned on a curated dataset with class balancing, augmentation, and weighted loss.
- Multi-view prediction with post-processing to mitigate out-of-distribution samples and common confusions.
- Nutrition database for every supported dish plus AI generated dietary guidance.
- Streamlit interface supporting multi-dish detection, manual corrections, and personalised user profiles.
- Utility scripts for dataset analysis and aggressive augmentation to bring minority classes up to a fixed target size.

## Repository Layout
```
├─ app/                # Streamlit UI entry point and cached feedback
├─ data/               # Train/val splits plus user feedback folders
├─ src/                # Core prediction + nutrition lookup modules
├─ augment_minority_classes.py
├─ data_analysis.py
├─ train_model.py      # Primary training script
├─ train_model_backup.py
├─ requirements.txt
└─ azeri_food_model.pt # Latest trained checkpoint (if present)
```

## Environment Setup
1. Install Python 3.9+.
2. Optionally create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. If you plan to use AI explanations, create a `.env` file with:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Dataset Preparation
- Place training images under `data/train/<class_name>` and validation images under `data/val/<class_name>`.
- Include a `non_meal` class populated with non-food images (people, landscapes, etc.) so the model can explicitly learn when no meal is present.
- Run the dataset report to understand balance and counts:
  ```bash
  python data_analysis.py --data_dir data/train
  ```
- To automatically expand all classes below a specific threshold to 150 images (default), run:
  ```bash
  python augment_minority_classes.py --data_dir data/train
  ```
  Use `--target` to change the goal or `--classes name1 name2` to restrict augmentation.

## Training
Train a new checkpoint with EfficientNet-B0 and weighted loss:
```bash
python train_model.py \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 25 \
  --batch_size 32
```
The script prints device information, class weights, and validation metrics. Best checkpoints are stored under `azeri_food_model.pt` unless configured otherwise.

## Prediction (CLI)
Use the lightweight predictor for single images:
```bash
python src/predict.py /path/to/image.jpg
```
Add `--single` to disable multi-dish detection. The script loads `azeri_food_model.pt` by default; pass `--checkpoint` if you want a different file.

## Streamlit Application
Launch the full UI with:
```bash
streamlit run app/ui.py
```
The app lets you upload photos, adjust detected dishes/portions, view aggregated macros, and optionally obtain AI explanations using the OpenAI API key.

## Nutrition Explanations
`src/genai_explainer.py` interfaces with the OpenAI Chat Completions API. Ensure the `.env` file includes `OPENAI_API_KEY`, and the `openai` package is installed. The Streamlit app toggles this feature via the sidebar.

## Model History
`azeri_food_model_history.json` tracks metadata for previous runs such as accuracy, loss, and hyperparameters. Update this file manually or extend training scripts to append entries after each experiment.

## Troubleshooting
- **Class not detected**: confirm images exist in `data/train/<class>` and rerun `python augment_minority_classes.py`.
- **CUDA/MPS errors**: verify that PyTorch detects your accelerator; otherwise the scripts fall back to CPU.
- **OpenAI errors**: ensure the API key is valid and network access is permitted.

## Contributing
1. Fork the repository and create a feature branch.
2. Add or update tests/analysis scripts where relevant.
3. Open a pull request with a focused description of changes, highlighting dataset or model updates when applicable.
