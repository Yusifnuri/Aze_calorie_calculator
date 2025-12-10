# genai_explainer.py
#
# Provides an AI-generated explanation of nutrition information
# using the OpenAI API (v1.x+ syntax).
#
# Requires environment variable:
#   export OPENAI_API_KEY="your_api_key_here"

from dotenv import load_dotenv
import os
from typing import Dict, Optional
from openai import OpenAI

load_dotenv()  # Load environment variables from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation(
    dish: str,
    nutrition: Dict,
    profile: Optional[Dict] = None,
) -> str:
    """
    Generate a short, personalised nutrition analysis for a predicted dish.

    Parameters
    ----------
    dish : str
        The recognized dish name.
    nutrition : dict
        Contains calories, fat, carbs, protein (numeric).
    profile : dict | None
        Optional user profile, e.g. goal, activity, diet preferences.

    Returns
    -------
    str
        AI-generated explanation.
    """

    profile_text = ""
    if profile:
        goal = profile.get("goal", "general health")
        activity = profile.get("activity", "moderate")
        diet_pref = profile.get("diet_preference", "no specific preference")
        profile_text = (
            f"\nUser goal: {goal}.\n"
            f"Activity level: {activity}.\n"
            f"Diet preference: {diet_pref}.\n"
        )

    prompt = f"""
You are a certified nutritionist. Analyze the Azerbaijani dish '{dish}'.

Nutritional information:
- Calories: {nutrition['calories']} kcal
- Fat: {nutrition['fat']} g
- Carbohydrates: {nutrition['carbs']} g
- Protein: {nutrition['protein']} g
{profile_text}
Write 5–7 sentences explaining:
1. Whether the dish is heavy/light/balanced in terms of energy.
2. Health implications of its calories, fat, carbs, and protein.
3. Suggest 2 healthier alternatives (optionally inspired by Azerbaijani cuisine).
4. Suggest 1 way to reduce calorie intake for this dish.
5. Adapt your advice to the user's goal, activity level and diet preference if provided.

Keep the explanation concise and easy to understand.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a nutrition expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content
