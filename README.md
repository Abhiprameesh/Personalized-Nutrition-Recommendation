# Personalized Nutrition Recommendation Engine

This repository contains a comprehensive **Personalized Nutrition Recommendation Engine** that leverages multiple machine learning models and nutritional datasets to provide highly customized diet plans based on an individual's genetics, regional baseline, lifestyle factors, and gender.

## Features

- **Regional Diet Prediction**: Estimates baseline caloric and macronutrient needs based on the user's country or region.
- **Genetic Sensitivity Profiling**: Uses Single Nucleotide Polymorphism (SNP) data to determine metabolic class (e.g., fat metabolism, protein metabolism) and adjust macronutrient needs accordingly.
- **Diet Adherence Scoring**: Evaluates lifestyle factors like sleep duration, physical activity, and stress levels to determine a diet adherence score, affecting the complexity of the recommended diet.
- **Gender-Specific Adjustments**: Calculates Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation and adjusts micro/macronutrients based on biological sex. It also re-ranks food recommendations (e.g., prioritizing iron-rich foods for females or high-protein foods for males).
- **Nutrient-based Food Similarity**: Uses `Cosine Similarity` to match the user's computed target nutritional profile against a dataset of foods to recommend the best options for each meal (Breakfast, Lunch, Dinner, Snack).

## Project Structure

- `recommendation_engine.py`: Core pipeline that loads models, processes user input, calculates targets, and uses cosine similarity to recommend foods.
- `gender_nutrient_adjustment.py`: Module for calculating BMR, daily caloric needs, and applying gender-specific food re-ranking.
- `test_recommendation.py`: Provides a complete end-to-end example of initializing the engine and running the pipeline on a mock user profile.
- `daily_food_nutrition_dataset.csv`: The core dataset containing foods, their categories, and their macronutrient profiles.
- `Genetic Sensitivity model/`: Contains the pre-trained genetic model (`genetic_model.pkl`) and necessary label encoders (`gene_encoder.pkl`, `snp_encoder.pkl`).
- `Regional Diet Prediction Model/`: Contains the pre-trained regional baseline model (`regional_diet_model.pkl`).
- `Diet adherence Model/`: Contains the pre-trained lifestyle adherence model (`diet_adherence_model.pkl`).

## Prerequisites

Ensure you have Python installed, along with the required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

You can run the full recommendation pipeline using the provided test script:

```bash
python test_recommendation.py
```

### Example Input Profile

The engine accepts a user profile containing various demographic, genetic, and lifestyle factors:

```python
user_input = {
    'Country': 'USA',
    'Sleep Duration': 6.5,
    'Physical Activity Level': 3,
    'Stress Level': 7,
    'Daily Steps': 8000,
    'SNP': 'rs9939609',
    'Gender': 'Female',
    'Age': 28,
    'Weight': 65,  # weight in kg
    'Height': 165  # height in cm
}
```

### Example Output Configuration

The engine evaluates this profile and returns:
1. **User Profile Analysis**: Identifies metabolic class and diet adherence score.
2. **Adjusted Nutrient Targets**: Calculates exact daily macronutrient and caloric targets tailored to the user's body.
3. **Optimized Meal Plan**: Suggests specific foods for each meal based on their similarity to the user's target nutrient profile, re-ranked based on gender-specific needs.
