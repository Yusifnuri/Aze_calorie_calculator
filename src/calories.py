# calories.py
#
# Approximate nutritional values per typical portion
# Units:
#   - calories: kcal
#   - fat: grams
#   - carbs: grams
#   - protein: grams

NUTRITION_TABLE = {
    # Azerbaijani dishes
    "plov": {"calories": 800, "fat": 32, "carbs": 95, "protein": 25},
    "dolma": {"calories": 350, "fat": 18, "carbs": 22, "protein": 17},
    "qutab": {"calories": 300, "fat": 12, "carbs": 32, "protein": 10},
    "lule_kabab": {"calories": 280, "fat": 20, "carbs": 4, "protein": 22},
    "shashlik": {"calories": 260, "fat": 18, "carbs": 3, "protein": 22},
    "dushbere": {"calories": 400, "fat": 14, "carbs": 48, "protein": 18},
    "bozbash": {"calories": 480, "fat": 24, "carbs": 32, "protein": 28},
    "piti": {"calories": 600, "fat": 28, "carbs": 40, "protein": 30},
    "xengel": {"calories": 550, "fat": 24, "carbs": 55, "protein": 20},
    "sac_kebab": {"calories": 700, "fat": 40, "carbs": 30, "protein": 35},
    "sekerbura": {"calories": 200, "fat": 11, "carbs": 22, "protein": 4},
    "paxlava": {"calories": 350, "fat": 22, "carbs": 34, "protein": 6},
    "yarpaq_xengel": {"calories": 500, "fat": 22, "carbs": 50, "protein": 18},
    "dovga": {"calories": 150, "fat": 6, "carbs": 16, "protein": 7},
    "seki_halvasi": {"calories": 250, "fat": 12, "carbs": 32, "protein": 4},
    "xash": {"calories": 350, "fat": 20, "carbs": 8, "protein": 30},
    "baki_qurabiyesi": {"calories": 180, "fat": 9, "carbs": 23, "protein": 3},
    "non_meal": {"calories": 0, "fat": 0, "carbs": 0, "protein": 0},

    # Global dishes
    "burger": {"calories": 550, "fat": 30, "carbs": 45, "protein": 25},
    "french_fries": {"calories": 350, "fat": 17, "carbs": 45, "protein": 4},
    "fried_chicken": {"calories": 400, "fat": 22, "carbs": 15, "protein": 30},
    "ice_cream": {"calories": 250, "fat": 14, "carbs": 28, "protein": 4},
    "pizza": {"calories": 700, "fat": 28, "carbs": 80, "protein": 28},
    "ramen": {"calories": 550, "fat": 20, "carbs": 70, "protein": 20},
    "spagetti": {"calories": 500, "fat": 12, "carbs": 80, "protein": 18},
    "sushi": {"calories": 300, "fat": 5, "carbs": 50, "protein": 15},
    "tiramisu": {"calories": 450, "fat": 28, "carbs": 45, "protein": 6},
}

# Quick calories lookup
CALORIE_TABLE = {dish: info["calories"] for dish, info in NUTRITION_TABLE.items()}
