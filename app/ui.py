# app/ui.py
#
# Streamlit UI with MULTI-DETECTION support:
#  - Automatically detects multiple dishes in one image
#  - Individual portion control for each dish
#  - Combined nutritional analysis
#  - AI-based personalised explanation

import sys
from io import BytesIO
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


st.set_page_config(
    page_title="FoodVisionAI - Azerbaijani Cuisine", 
    page_icon="🍽️",
    layout="wide"
)

st.title("🇦🇿 Azerbaijani Cuisine Nutrition Analyzer")

st.markdown("""
**Multi-Dish Detection AI** - Upload an image and the app will:
- 🔍 Detect **multiple dishes** automatically
- 📊 Calculate nutrition for each dish separately  
- ⚖️ Allow individual portion adjustments
- 🤖 Generate personalized AI nutrition insights
""")

# Sidebar: user profile + settings
with st.sidebar:
    st.header("👤 User Profile")

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

    st.divider()
    st.header("⚙️ Settings")
    
    multi_detection = st.checkbox(
        "Enable multi-dish detection",
        value=True,
        help="Automatically detect multiple dishes in one image"
    )
    
    use_ai_explanation = st.checkbox(
        "Enable AI nutrition explanation",
        value=True,
        help="Generate personalized insights using AI"
    )
    
    debug_mode = st.checkbox(
        "🔍 Debug mode",
        value=False,
        help="Show all predictions and confidence scores"
    )

# Build profile
profile = {
    "goal": goal,
    "activity": activity,
    "diet_preference": diet_pref,
}

# Keep last prediction to avoid rerunning model on every widget interaction
if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None
    st.session_state.last_prediction = None

# Main content
uploaded_image = st.file_uploader(
    "📸 Upload a food image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    
    with col2:
        should_run_inference = (
            st.session_state.last_image_bytes != image_bytes
            or st.session_state.last_prediction is None
        )

        if should_run_inference:
            with st.spinner("🔄 Analyzing image with AI..."):
                result = predict_dish_and_nutrition(
                    img, 
                    multi_detection=multi_detection
                )
            st.session_state.last_prediction = result
            st.session_state.last_image_bytes = image_bytes
        else:
            result = st.session_state.last_prediction

        detected_dishes = result["detected_dishes"]
        primary_dish = result["primary_dish"]
        is_confident = result["is_confident"]
        all_candidates = result.get("all_candidates", [])
        all_classes = result.get("all_classes", [])

        # ---------------------------------------------------------
        # CASE 1: Model is NOT confident / No meals detected
        # ---------------------------------------------------------
        if not is_confident or len(detected_dishes) == 0:
            if primary_dish is None:
                st.error("🚫 Not Food / Unknown")
                st.warning("Bu şəkil yemək kimi görünmür (və ya model qərarsızdır). Zəhmət olmasa yemək şəkli yüklə.")
            else:
                st.error("🚫 No Meals Detected")
                st.warning(
                    "The AI could not confidently identify any Azerbaijani dishes in this image. "
                    "This could mean:\n"
                    "- The image doesn't contain Azerbaijani cuisine\n"
                    "- The image quality is too low\n"
                    "- The dish is not in the training dataset"
                )
            
            if result['primary_confidence'] > 0:
                st.write(f"Best guess confidence: **{result['primary_confidence'] * 100:.2f}%** (below threshold)")
            
            if all_candidates:
                st.info("**Top uncertain candidates:**")
                for c in all_candidates[:5]:
                    st.write(f"- {c['dish']} ({c['confidence'] * 100:.1f}%)")

            st.divider()
            st.subheader("🔧 Manual Input")
            st.write("If you know what dish this is, you can enter it manually:")
            
            manual_name = st.text_input(
                "Dish name:",
                placeholder="e.g., Plov, Dolma, Kebab..."
            )

            if manual_name:
                # Check if dish exists in database
                if manual_name in NUTRITION_TABLE:
                    st.success(f"✅ Found '{manual_name}' in database!")
                    
                    nut = NUTRITION_TABLE[manual_name]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Calories", f"{nut['calories']}", "kcal")
                    with col2:
                        st.metric("Fat", f"{nut['fat']}", "g")
                    with col3:
                        st.metric("Carbs", f"{nut['carbs']}", "g")
                    with col4:
                        st.metric("Protein", f"{nut['protein']}", "g")
                    
                    if use_ai_explanation:
                        with st.spinner("Generating AI explanation..."):
                            explanation = generate_explanation(
                                manual_name,
                                nut,
                                profile=profile,
                            )
                        st.info(explanation)
                else:
                    st.warning(f"⚠️ '{manual_name}' not found in nutrition database.")
                    st.write("**Available dishes:**")
                    st.write(", ".join(sorted(all_classes[:20])) + "...")
            
            st.stop()
        
        # ---------------------------------------------------------
        # CASE 2: Model is confident - show detections
        # ---------------------------------------------------------
        
        if multi_detection and len(detected_dishes) > 1:
            st.success(f"✅ Detected **{len(detected_dishes)}** dishes in the image!")
        else:
            st.success(f"✅ Detected: **{primary_dish}**")
        
        # Debug mode - show all top predictions
        if debug_mode:
            st.warning("🔍 **DEBUG MODE** - Top 10 predictions:")
            debug_df = []
            for i, c in enumerate(all_candidates[:10], 1):
                debug_df.append({
                    "Rank": i,
                    "Dish": c['dish'],
                    "Confidence": f"{c['confidence']*100:.2f}%"
                })
            st.dataframe(debug_df, hide_index=True, use_container_width=True)
            
            if len(detected_dishes) > 0:
                st.info(f"✅ After filtering: {len(detected_dishes)} dish(es) passed threshold")

    st.divider()
    
    # ---------------------------------------------------------
    # Display detected dishes with individual portion control
    # ---------------------------------------------------------
    st.header("🍽️ Detected Dishes")
    
    # Initialize session state for portions if not exists
    if 'portions' not in st.session_state:
        st.session_state.portions = {}
    
    # Display each detected dish
    dish_selections = []
    
    for i, dish_info in enumerate(detected_dishes):
        dish_name = dish_info["dish"]
        confidence = dish_info["confidence"]
        is_primary = dish_info["is_primary"]
        nutrition = dish_info["nutrition"]
        
        # Create expandable section for each dish
        with st.expander(
            f"{'🍽️ ' if is_primary else '  '}{dish_name} ({confidence*100:.1f}%)", 
            expanded=(i < 2)
        ):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Include/exclude checkbox
                include = st.checkbox(
                    "Include in total",
                    value=True,
                    key=f"include_{dish_name}_{i}"
                )
            
            with col2:
                # Portion slider
                portion = st.slider(
                    "Portion",
                    min_value=0.25,
                    max_value=3.0,
                    value=1.0,
                    step=0.25,
                    key=f"portion_{dish_name}_{i}",
                    help="0.5 = half portion, 1.0 = standard, 2.0 = double"
                )
            
            with col3:
                st.metric("Confidence", f"{confidence*100:.0f}%")
            
            # Show nutrition for this dish
            st.markdown("**Nutrition (per portion):**")
            
            ncol1, ncol2, ncol3, ncol4 = st.columns(4)
            with ncol1:
                calories = nutrition['calories'] * portion
                st.metric("Calories", f"{calories:.0f}", "kcal")
            with ncol2:
                fat = nutrition['fat'] * portion
                st.metric("Fat", f"{fat:.1f}", "g")
            with ncol3:
                carbs = nutrition['carbs'] * portion
                st.metric("Carbs", f"{carbs:.1f}", "g")
            with ncol4:
                protein = nutrition['protein'] * portion
                st.metric("Protein", f"{protein:.1f}", "g")
            
            if include:
                dish_selections.append({
                    "name": dish_name,
                    "portion": portion,
                    "nutrition": nutrition
                })
    
    # ---------------------------------------------------------
    # Manual correction option
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🔧 Manual Correction (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Add additional dish:**")
        additional_dish = st.selectbox(
            "Select from all available dishes",
            options=["None"] + all_classes,
            index=0
        )
        
        if additional_dish != "None":
            additional_portion = st.slider(
                "Portion size",
                min_value=0.25,
                max_value=3.0,
                value=1.0,
                step=0.25,
                key="additional_portion"
            )
            
            if st.button("➕ Add to meal"):
                if additional_dish in NUTRITION_TABLE:
                    dish_selections.append({
                        "name": additional_dish,
                        "portion": additional_portion,
                        "nutrition": NUTRITION_TABLE[additional_dish]
                    })
                    st.success(f"Added {additional_dish} x{additional_portion}")
    
    with col2:
        st.write("**Correct primary dish:**")
        main_dish_default_idx = all_classes.index(primary_dish) if primary_dish in all_classes else 0
        corrected_main_dish = st.selectbox(
            "If the main dish is wrong, select correct one:",
            options=all_classes,
            index=main_dish_default_idx
        )
    
    # ---------------------------------------------------------
    # Calculate combined nutrition
    # ---------------------------------------------------------
    st.divider()
    st.header("📊 Total Nutritional Information")
    
    combined_nutrition = {
        "calories": 0.0, 
        "fat": 0.0, 
        "carbs": 0.0, 
        "protein": 0.0
    }
    
    for dish_sel in dish_selections:
        nut = dish_sel["nutrition"]
        portion = dish_sel["portion"]
        combined_nutrition["calories"] += nut["calories"] * portion
        combined_nutrition["fat"] += nut["fat"] * portion
        combined_nutrition["carbs"] += nut["carbs"] * portion
        combined_nutrition["protein"] += nut["protein"] * portion
    
    # Display totals
    st.markdown("### Complete Meal Summary")
    
    selected_dish_names = [d["name"] for d in dish_selections]
    st.write(f"**Dishes:** {', '.join(selected_dish_names) if selected_dish_names else 'None selected'}")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔥 Total Calories", 
            f"{combined_nutrition['calories']:.0f}",
            delta=None,
            help="Total energy from all dishes"
        )
    
    with col2:
        st.metric(
            "🥑 Total Fat", 
            f"{combined_nutrition['fat']:.1f} g",
            delta=None
        )
    
    with col3:
        st.metric(
            "🍞 Total Carbs", 
            f"{combined_nutrition['carbs']:.1f} g",
            delta=None
        )
    
    with col4:
        st.metric(
            "🥩 Total Protein", 
            f"{combined_nutrition['protein']:.1f} g",
            delta=None
        )
    
    # Macronutrient breakdown
    st.markdown("### Macronutrient Distribution")
    
    total_macros = combined_nutrition['fat'] + combined_nutrition['carbs'] + combined_nutrition['protein']
    
    if total_macros > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fat_pct = (combined_nutrition['fat'] / total_macros) * 100
            st.progress(fat_pct / 100)
            st.write(f"Fat: {fat_pct:.1f}%")
        
        with col2:
            carbs_pct = (combined_nutrition['carbs'] / total_macros) * 100
            st.progress(carbs_pct / 100)
            st.write(f"Carbs: {carbs_pct:.1f}%")
        
        with col3:
            protein_pct = (combined_nutrition['protein'] / total_macros) * 100
            st.progress(protein_pct / 100)
            st.write(f"Protein: {protein_pct:.1f}%")
    
    # ---------------------------------------------------------
    # AI Explanation
    # ---------------------------------------------------------
    st.divider()
    st.header("🤖 AI Nutrition Analysis")
    
    if use_ai_explanation and combined_nutrition['calories'] > 0:
        with st.spinner("Generating personalized AI insights..."):
            # Create meal description
            meal_description = ", ".join(selected_dish_names)
            
            explanation = generate_explanation(
                meal_description,
                combined_nutrition,
                profile=profile,
            )
        
        st.info(explanation)
    elif not use_ai_explanation:
        st.info("🔒 AI explanation is disabled. Enable it in the sidebar for personalized insights.")
    else:
        st.warning("⚠️ No dishes selected. Please include at least one dish for AI analysis.")
    
    # ---------------------------------------------------------
    # User Feedback
    # ---------------------------------------------------------
    st.divider()
    st.header("💬 Feedback")
    
    feedback_col1, feedback_col2 = st.columns([3, 1])
    
    with feedback_col1:
        feedback = st.radio(
            "Was the detection accurate?",
            ["✅ Yes, accurate", "❌ No, needs improvement"],
            index=0,
            horizontal=True
        )
    
    if feedback == "❌ No, needs improvement":
        st.write("**Help us improve by providing the correct labels:**")
        
        feedback_dishes = st.multiselect(
            "What dishes are actually in the image?",
            options=all_classes,
            default=[]
        )
        
        if st.button("📤 Submit Feedback"):
            import uuid
            
            # Save image with feedback metadata
            feedback_dir = Path("data/user_feedback/multi_dish")
            feedback_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = feedback_dir / filename
            
            img.save(save_path)
            
            # Save metadata
            metadata_path = feedback_dir / f"{uuid.uuid4().hex}_metadata.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Detected: {', '.join([d['dish'] for d in detected_dishes])}\n")
                f.write(f"Actual: {', '.join(feedback_dishes)}\n")
            
            st.success(f"✅ Feedback saved! Thank you for helping improve the model.")
else:
    st.session_state.last_image_bytes = None
    st.session_state.last_prediction = None
    # Landing state
    st.info("👆 Upload an image to get started")
    
    st.markdown("""
    ### How it works:
    
    1. **Upload** a photo of Azerbaijani cuisine
    2. **AI detects** all dishes automatically (multi-dish support!)
    3. **Adjust** portions for each dish individually
    4. **View** complete nutritional breakdown
    5. **Get** personalized AI insights based on your goals
    
    ---
    
    #### Supported features:
    - ✅ Multi-dish detection in single image
    - ✅ Individual portion control per dish
    - ✅ Real-time nutrition calculation
    - ✅ AI-powered personalized recommendations
    - ✅ Manual corrections and additions
    - ✅ User feedback collection
    """)
