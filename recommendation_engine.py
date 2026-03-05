import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

class FoodRecommendationEngine:
    def __init__(self, dataset_path, model_paths):
        """
        Initialize the recommendation engine by loading the food dataset and the ML models.
        
        :param dataset_path: Path to the daily_food_nutrition_dataset.csv
        :param model_paths: Dictionary containing paths to the 3 `.pkl` models and encoders
        """
        # 1. Load the food dataset
        self.food_df = pd.read_csv(dataset_path, on_bad_lines='skip')
        
        # 2. Extract and Normalize nutrient features
        self.nutrient_features = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)']
        # Only keep foods with non-null values in these features
        self.food_df = self.food_df.dropna(subset=self.nutrient_features).copy()
        
        self.scaler = StandardScaler()
        # Create the normalized nutrient matrix
        self.nutrient_matrix = self.scaler.fit_transform(self.food_df[self.nutrient_features])
        
        # 3. Load the ML models
        self.genetic_model = joblib.load(model_paths.get('genetic_model'))
        # Try to load encoders if they exist (based on earlier search)
        self.gene_encoder = None
        self.snp_encoder = None
        if 'gene_encoder' in model_paths and os.path.exists(model_paths['gene_encoder']):
            self.gene_encoder = joblib.load(model_paths['gene_encoder'])
        if 'snp_encoder' in model_paths and os.path.exists(model_paths['snp_encoder']):
            self.snp_encoder = joblib.load(model_paths['snp_encoder'])
            
        self.regional_model = joblib.load(model_paths.get('regional_diet_model'))
        self.adherence_model = joblib.load(model_paths.get('diet_adherence_model'))

    def predict_user_profile(self, user_input):
        """
        Uses the three models to predict baseline nutrition, metabolic sensitivity, and adherence.
        """
        # --- 1. Regional Diet Baseline ---
        # Note: The expected features depend heavily on how the regional model was trained.
        # We assume it takes 'Country' (maybe encoded) or similar. 
        # For a robust script, we will pass a dataframe with the required features.
        # If 'Country' is categorical, the model pipeline should ideally handle it,
        # otherwise we might need a specific encoder mapping.
        
        # In this implementation, we construct DataFrames for the models.
        # This will need adjustment if the exact features expected by the models differ.
        try:
            regional_features = pd.DataFrame([{
                'Country': user_input.get('Country', 'Unknown')
            }])
            # Add missing columns or use directly if pipeline handles it
            brand_new_regional = self.regional_model.predict(regional_features)
            # Assuming output is [Protein, Fat, Calories] or similar based on Prompt:
            # Predicts baseline dietary intake values for a given country:
            # - Protein intake (g/day)
            # - Fat intake (g/day)
            # - Calories (kcal/day)
            # Output might be a 1D array of length 3 or a 2D array [[Protein, Fat, Calories]]
            if len(brand_new_regional.shape) > 1:
                baseline_vals = brand_new_regional[0]
            else:
                baseline_vals = brand_new_regional
            
            baseline = {
                'protein': baseline_vals[0] if len(baseline_vals) > 0 else 50,
                'fat': baseline_vals[1] if len(baseline_vals) > 1 else 70,
                'calories': baseline_vals[2] if len(baseline_vals) > 2 else 2000
            }
        except Exception as e:
            print(f"Warning: Regional model prediction failed ({e}). Using default baselines.")
            baseline = {'protein': 60, 'fat': 70, 'calories': 2000}


        # --- 2. Genetic Sensitivity ---
        try:
            snp_val = user_input.get('SNP', 'Unknown')
            if self.snp_encoder:
                 snp_encoded = self.snp_encoder.transform([[snp_val]])[0]
            else:
                 snp_encoded = snp_val
                 
            genetic_features = pd.DataFrame([{
                'SNP': snp_encoded
            }])
            gen_pred = self.genetic_model.predict(genetic_features)
            metabolic_class = gen_pred[0]
            
            # Decode if necessary
            if self.gene_encoder and isinstance(metabolic_class, (int, np.integer)):
                 metabolic_class = self.gene_encoder.inverse_transform([metabolic_class])[0]

        except Exception as e:
             print(f"Warning: Genetic model prediction failed ({e}). Defaulting to normal_metabolism.")
             metabolic_class = 'normal_metabolism'


        # --- 3. Diet Adherence Score ---
        try:
            adherence_features = pd.DataFrame([{
                'Sleep Duration': user_input.get('Sleep Duration', 7),
                'Physical Activity Level': user_input.get('Physical Activity Level', 2),
                'Stress Level': user_input.get('Stress Level', 5),
                'Daily Steps': user_input.get('Daily Steps', 5000)
            }])
            adh_pred = self.adherence_model.predict(adherence_features)
            adherence_score = float(adh_pred[0])
        except Exception as e:
            print(f"Warning: Adherence model prediction failed ({e}). Defaulting to 0.5.")
            adherence_score = 0.5
            
        return baseline, metabolic_class, adherence_score

    def compute_target_nutrition(self, baseline, metabolic_class, adherence_score):
        """
        Adjust target macro/micronutrients based on genetic sensitivities and adherence.
        """
        # Start with baseline (per Day targets)
        calories = baseline.get('calories', 2000)
        protein = baseline.get('protein', 60)
        fat = baseline.get('fat', 70)
        
        # Estimate Carbohydrates based on remaining calories
        # Calories = (Protein * 4) + (Fat * 9) + (Carbs * 4)
        carbs = (calories - (protein * 4) - (fat * 9)) / 4
        if carbs < 0:
            carbs = 250 # fallback
            
        fiber = 30 # default healthy fiber target

        # Genetic adjustments based on metabolic_class
        if metabolic_class == 'fat_metabolism':
            # Reduce fat intake
            fat *= 0.8
            # Compensate with carbs
            carbs += ((baseline.get('fat', 70) * 0.2) * 9) / 4
        elif metabolic_class == 'protein_metabolism':
            # Increase/adjust protein intake
            protein *= 1.2
            # Reduce carbs to compensate
            carbs -= ((baseline.get('protein', 60) * 0.2) * 4) / 4
        elif metabolic_class == 'vitamin_metabolism':
            # Prioritize vitamins - represented here by higher fiber target (e.g. fruits/veg)
            fiber *= 1.5
            
        # Adherence adjustments
        # While complexity isn't a macronutrient, we can use adherence_score later 
        # to filter the kinds of foods, but we'll store it here abstractly.
        complexity = 'high' if adherence_score > 0.7 else ('low' if adherence_score < 0.4 else 'medium')

        # Calculate Per-Meal Targets (Assuming 3 main meals and 1 snack)
        # Distribution: Breakfast 25%, Lunch 35%, Dinner 30%, Snack 10%
        meal_distributions = {
            'Breakfast': 0.25,
            'Lunch': 0.35,
            'Dinner': 0.30,
            'Snack': 0.10
        }
        
        daily_targets = {
            'Calories (kcal)': calories,
            'Protein (g)': protein,
            'Carbohydrates (g)': carbs,
            'Fat (g)': fat,
            'Fiber (g)': fiber
        }

        target_profiles = {}
        for meal, dist in meal_distributions.items():
             target_profiles[meal] = {k: v * dist for k, v in daily_targets.items()}
             
        return target_profiles, complexity

    def recommend_foods(self, target_profiles, complexity, top_n=5):
        """
        Use cosine similarity to find top matched foods for each meal type.
        """
        recommendations = {}
        
        for meal_type, target_vec_dict in target_profiles.items():
             # Create vector for the current meal target
             target_vec = [
                 target_vec_dict['Calories (kcal)'],
                 target_vec_dict['Protein (g)'],
                 target_vec_dict['Carbohydrates (g)'],
                 target_vec_dict['Fat (g)'],
                 target_vec_dict['Fiber (g)']
             ]
             
             # Transform target using the fitted scaler
             target_normalized = self.scaler.transform([target_vec])
             
             # Filter dataset by Meal_Type if the dataset has meals classified
             # Note: Meal_Type might contain multiple tags like "Breakfast/Snack", so we use str.contains
             meal_mask = self.food_df['Meal_Type'].str.contains(meal_type, case=False, na=False)
             
             # If complexity is low, we might filter out 'Processed' or 'Complex' meals if possible,
             # but keeping it simple based on the instructions:
             
             filtered_df = self.food_df[meal_mask]
             if filtered_df.empty:
                 # Fallback to all foods if no specific meal matches
                 filtered_df = self.food_df
                 
             # Get indices in the original dataframe
             filtered_indices = filtered_df.index
             
             # Extract the subset of the normalized matrix for the filtered foods
             # This requires knowing their row index in the full nutrient_matrix
             # A safer way is to re-extract for the filtered df to avoid index mismatch
             filtered_matrix_subset = self.nutrient_matrix[self.food_df.index.isin(filtered_indices)]
             
             # Compute Cosine Similarity between target_normalized and filtered foods
             similarities = cosine_similarity(target_normalized, filtered_matrix_subset)[0]
             
             # Get top N indices
             top_indices_local = similarities.argsort()[-top_n:][::-1]
             
             meal_recs = []
             seen_foods = set()
             
             # Map back to original dataframe via iloc on filtered_df
             for idx in top_indices_local:
                 food_row = filtered_df.iloc[idx]
                 food_name = food_row['Food_Item']
                 
                 # Skip if we already recommended this exact item for this meal
                 if food_name in seen_foods:
                     continue
                 seen_foods.add(food_name)
                 
                 sim_score = similarities[idx]
                 
                 meal_recs.append({
                     'Food_Item': food_name,
                     'Category': food_row['Category'],
                     'Calories (kcal)': food_row['Calories (kcal)'],
                     'Protein (g)': food_row['Protein (g)'],
                     'Carbohydrates (g)': food_row['Carbohydrates (g)'],
                     'Fat (g)': food_row['Fat (g)'],
                     'Similarity Score': round(float(sim_score), 4)
                 })
                 
                 if len(meal_recs) >= top_n:
                     break
                 
             recommendations[meal_type] = meal_recs
        
        return recommendations

    def generate_diet_plan(self, user_input, top_n=3):
        """
        End-to-end pipeline: Process user input, get predictions, compute targets, and return structured recommendations.
        """
        # 1. Predict
        baseline, metabolic_class, adherence_score = self.predict_user_profile(user_input)
        
        # 2. Compute Target Adjustments
        target_profiles, complexity = self.compute_target_nutrition(baseline, metabolic_class, adherence_score)
        
        print("\n--- User Profile Analysis ---")
        print(f"Metabolic Class: {metabolic_class}")
        print(f"Adherence Score: {adherence_score:.2f} ({complexity} complexity)")
        print(f"Baseline: {baseline}")
        
        # 3. Recommend Foods
        recommendations = self.recommend_foods(target_profiles, complexity, top_n)
        
        return recommendations

