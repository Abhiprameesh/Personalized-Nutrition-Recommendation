import os
from recommendation_engine import FoodRecommendationEngine
from gender_nutrient_adjustment import GenderNutrientAdjustment

def test_recommendation_pipeline():
    # 1. Define paths to models and dataset
    base_dir = r"c:\ALLMLPROJ\Personalized-Nutrition"
    dataset_path = os.path.join(base_dir, "daily_food_nutrition_dataset.csv")
    
    model_paths = {
        'genetic_model': os.path.join(base_dir, 'Genetic Sensitivity model', 'genetic_model.pkl'),
        'gene_encoder': os.path.join(base_dir, 'Genetic Sensitivity model', 'gene_encoder.pkl'),
        'snp_encoder': os.path.join(base_dir, 'Genetic Sensitivity model', 'snp_encoder.pkl'),
        'regional_diet_model': os.path.join(base_dir, 'Regional Diet Prediction Model', 'regional_diet_model.pkl'),
        'diet_adherence_model': os.path.join(base_dir, 'Diet adherence Model', 'diet_adherence_model.pkl')
    }
    
    # 2. Initialize the engine
    print("Initializing Food Recommendation Engine...")
    try:
        engine = FoodRecommendationEngine(dataset_path, model_paths)
        print("Engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # 3. Create a mock user profile with gender attributes
    user_input = {
        'Country': 'USA',
        'Sleep Duration': 6.5,
        'Physical Activity Level': 3,
        'Stress Level': 7,
        'Daily Steps': 8000,
        'SNP': 'rs9939609',
        'Gender': 'Female',
        'Age': 28,
        'Weight': 65,
        'Height': 165
    }
    
    # 4. Generate Initial Recommendations (Top 10 so gender module has variety to re-rank)
    print("\nGenerating Initial Diet Plan...")
    initial_diet_plan = engine.generate_diet_plan(user_input, top_n=10)
    
    # 5. Apply Gender Nutrient Adjustments
    print("\nApplying Gender-Specific Adjustments...")
    gender_adjuster = GenderNutrientAdjustment()
    final_plan = gender_adjuster.process(user_input, initial_diet_plan, meal_top_n=3)

    print("\n--- Adjusted Nutrient Targets ---")
    print(f"Calories: {final_plan['adjusted_calorie_target']} kcal")
    print(f"Macronutrients: {final_plan['macronutrient_targets']}")
    print(f"Micronutrients: {final_plan['micronutrient_targets']}")
    
    # 6. Output final recommendations
    print("\n--- Final Recommended Diet Plan ---")
    for meal, foods in final_plan['adjusted_meal_plan'].items():
        print(f"\n{meal.upper()}:")
        for i, food in enumerate(foods, 1):
             print(f"  {i}. {food['Food_Item']} ({food['Category']}) - Similarity: {food['Similarity Score']}")
             print(f"     Nutrients: {food['Calories (kcal)']} kcal, Prot: {food['Protein (g)']}g, "
                   f"Carbs: {food['Carbohydrates (g)']}g, Fat: {food['Fat (g)']}g")


if __name__ == "__main__":
    test_recommendation_pipeline()
