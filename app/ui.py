# app/ui.py
#
# Streamlit UI for:
#  - uploading an image
#  - predicting Azerbaijani dish
#  - showing nutritional values
#  - multiple-food selection & portion-size estimation
#  - AI-based personalised nutrition explanation

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make sure we can import from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_dish_and_nutrition
from genai_explainer import generate_explanation
from calories import NUTRITION_TABLE


st.set_page_config(page_title="FoodVisionAI - Azerbaijani Cuisine", page_icon="🍽️")

st.title("Azerbaijani Cuisine Nutrition Analyzer 🇦🇿")

st.write(
    """
Upload an image of an Azerbaijani dish and the app will:

1. Recognize the dish using a fine-tuned EfficientNet-B0 model  
2. Estimate its nutritional values (calories, fat, carbs, protein)  
3. Let you correct or refine the detection (multiple dishes, portion size)  
4. Generate an AI-based, personalised nutrition explanation
    """
)

# Sidebar: user profile + AI
with st.sidebar:
    st.header("User profile (optional)")

    goal = st.selectbox(
        "Goal",
        ["General health", "Weight loss", "Muscle gain"],
        index=0,
    )
    activity = st.selectbox(
        "Activity level",
        ["Low", "Moderate", "High"],
        index=1,
    )
    diet_pref = st.selectbox(
        "Diet preference",
        ["No specific preference", "Vegetarian", "Low-carb"],
        index=0,
    )

    use_ai_explanation = st.checkbox(
        "Enable AI nutrition explanation (uses API)",
        value=True,
    )

# Build profile once so we can use it in both normal and low-confidence cases
profile = {
    "goal": goal,
    "activity": activity,
    "diet_preference": diet_pref,
}

uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing image with FoodVisionAI..."):
        result = predict_dish_and_nutrition(img)

    # Extract prediction results
    dish = result["dish"]
    confidence = result["confidence"]
    nutrition = result["nutrition"]
    top_candidates = result.get("top_candidates", [])
    all_classes = result.get("all_classes", [])

    # ---------------------------------------------------------
    # CASE 1: The model is NOT confident enough
    # ---------------------------------------------------------
    if dish is None:
        st.subheader("Model prediction")
        st.error("The model could not confidently identify any dish in the image.")
        st.info(f"Confidence score of the best guess: {confidence * 100:.2f}% (below threshold)")

        # Show the top candidates so the user sees what the model roughly considered
        if top_candidates:
            st.write("Top candidates (low confidence):")
            for c in top_candidates:
                st.write(f"- {c['dish']} ({c['confidence'] * 100:.1f}%)")

        # Allow the user to manually enter the dish name
        st.subheader("Manual dish input")
        manual_name = st.text_input(
           "If you know the dish name, you can enter it manually:"
        )

        # Optional: AI explanation even in low-confidence case
        if manual_name and use_ai_explanation:
            with st.spinner("Generating AI explanation based on your input..."):
                explanation = generate_explanation(
                    manual_name,
                    None,          # no reliable nutrition estimate from the model
                    profile=profile,
                )
            st.write(explanation)
        elif manual_name and not use_ai_explanation:
            st.info(
                "AI explanation is disabled (speed/cost optimisation). "
                "Enable it in the sidebar if you want a personalised analysis."
            )

        # Stop the rest of the UI flow since we don't trust the prediction
        st.stop()

    # ---------------------------------------------------------
    # CASE 2: Model is confident enough → normal flow
    # ---------------------------------------------------------

    # Fallbacks if keys are missing
    if not top_candidates:
        top_candidates = [{"dish": dish, "confidence": confidence}]
    if not all_classes:
        all_classes = [dish]

    # ---- Prediction summary ----
    st.subheader("Model prediction")
    st.write(f"**Top-1 predicted dish:** {dish}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # ---- Multiple-food detection (soft version via top-k candidates) ----
    st.subheader("Detected dishes (top candidates)")

    candidate_labels = [
        f"{c['dish']} ({c['confidence'] * 100:.1f}%)" for c in top_candidates
    ]
    default_selection = [candidate_labels[0]] if candidate_labels else []

    selected_labels = st.multiselect(
        "Select the dishes that are actually present in the image "
        "(you can choose multiple):",
        options=candidate_labels,
        default=default_selection,
    )

    # Map selected labels back to dish names
    selected_dishes = []
    for label in selected_labels:
        idx = candidate_labels.index(label)
        selected_dishes.append(top_candidates[idx]["dish"])

    # ---- Manual correction of main dish label ----
    st.subheader("Main dish correction (optional)")

    main_dish_default_idx = all_classes.index(dish) if dish in all_classes else 0
    corrected_main_dish = st.selectbox(
        "If the main dish is wrong, select the correct one:",
        options=all_classes,
        index=main_dish_default_idx,
    )

    # If user did not select any dishes in multiselect, fall back to main dish
    if not selected_dishes:
        selected_dishes = [corrected_main_dish]

    # ---- Portion-size estimation ----
    st.subheader("Portion-size estimation")

    portion_factor = st.slider(
        "Portion factor (relative to one standard portion per dish)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="0.5 = half portion, 1.0 = one portion, 2.0 = double portion, etc.",
    )

    # Combine nutrition across selected dishes * portion_factor
    combined_nutrition = {"calories": 0.0, "fat": 0.0, "carbs": 0.0, "protein": 0.0}

    for d in selected_dishes:
        if d not in NUTRITION_TABLE:
            continue
        info = NUTRITION_TABLE[d]
        combined_nutrition["calories"] += info["calories"] * portion_factor
        combined_nutrition["fat"] += info["fat"] * portion_factor
        combined_nutrition["carbs"] += info["carbs"] * portion_factor
        combined_nutrition["protein"] += info["protein"] * portion_factor

    st.write(
        f"Selected dishes: {', '.join(selected_dishes)} "
        f"(portion factor: x{portion_factor:.1f})"
    )

    # ---- Save user feedback ----
st.subheader("Was the prediction correct?")

feedback = st.radio(
    "Help improve the model:",
    ["Yes, prediction was correct", "No, the model was wrong"],
    index=0,
)

if feedback == "No, the model was wrong":
    st.write("Select the correct dish label:")
    correct_label = st.selectbox("Correct dish:", options=all_classes)

    if st.button("Save feedback"):
        import os
        import uuid

        feedback_dir = Path("data/user_feedback") / correct_label
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = feedback_dir / filename

        img.save(save_path)
        st.success(f"Feedback saved! Image stored as {save_path}")


    # ---- Nutritional information output ----
    st.subheader("Estimated nutritional information (total)")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Calories", f"{combined_nutrition['calories']:.0f} kcal")
        st.metric("Fat", f"{combined_nutrition['fat']:.1f} g")

    with col2:
        st.metric("Carbs", f"{combined_nutrition['carbs']:.1f} g")
        st.metric("Protein", f"{combined_nutrition['protein']:.1f} g")

    # ---- AI-based personalised explanation ----
    st.subheader("AI Nutrition Analysis")

    if use_ai_explanation:
        with st.spinner("Generating AI explanation..."):
            explanation = generate_explanation(
                corrected_main_dish,
                {
                    "calories": combined_nutrition["calories"],
                    "fat": combined_nutrition["fat"],
                    "carbs": combined_nutrition["carbs"],
                    "protein": combined_nutrition["protein"],
                },
                profile=profile,
            )
        st.write(explanation)
    else:
        st.info(
            "AI explanation is disabled (speed/cost optimisation). "
            "Enable it in the sidebar if you want a personalised analysis."
        )
