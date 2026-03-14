import streamlit as st
import pandas as pd
import os
from recommendation_engine import FoodRecommendationEngine
from gender_nutrient_adjustment import GenderNutrientAdjustment

st.set_page_config(
    page_title="Personalized Nutrition Engine",
    page_icon="🍲",
    layout="wide"
)

st.title("🍲 Personalized Nutrition Recommendation Engine")
st.markdown("""
Welcome to the multi-model **Personalized Nutrition Engine**. 
This system uses Data Dependency Analysis across your genetics, regional baseline, and lifestyle factors to generate a truly personalized diet plan.
""")

# --- Sidebar Inputs ---
st.sidebar.header("User Profile Setup")

with st.sidebar.form("user_input_form"):
    st.subheader("Demographics")
    gender = st.radio("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=15, max_value=100, value=28)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0)
    height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=165.0)
    country = st.selectbox("Country/Region", ["USA", "India", "UK", "Canada", "Australia", "Brazil", "China", "France"])
    
    st.subheader("Genetic Profile")
    snp = st.text_input("Primary SNP Marker", value="rs9939609", help="e.g. rs9939609")
    
    st.subheader("Lifestyle & Habits")
    sleep = st.slider("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=6.5, step=0.5)
    activity = st.slider("Physical Activity Level (0-100)", min_value=0, max_value=100, value=30)
    stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=7)
    steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000, step=500)
    
    submit_button = st.form_submit_button(label="Generate Recommendations")

# --- Main Logic ---
if submit_button:
    user_input = {
        'Country': country,
        'Sleep Duration': sleep,
        'Physical Activity Level': activity,
        'Stress Level': stress,
        'Daily Steps': steps,
        'SNP': snp,
        'Gender': gender,
        'Age': age,
        'Weight': weight,
        'Height': height
    }
    
    with st.spinner("Analyzing Genetic, Regional, and Lifestyle Models..."):
        try:
            base_dir = r"c:\ALLMLPROJ\Personalized-Nutrition"
            dataset_path = os.path.join(base_dir, "daily_food_nutrition_dataset.csv")
            
            model_paths = {
                'genetic_model': os.path.join(base_dir, 'Genetic Sensitivity model', 'genetic_model.pkl'),
                'gene_encoder': os.path.join(base_dir, 'Genetic Sensitivity model', 'gene_encoder.pkl'),
                'snp_encoder': os.path.join(base_dir, 'Genetic Sensitivity model', 'snp_encoder.pkl'),
                'regional_diet_model': os.path.join(base_dir, 'Regional Diet Prediction Model', 'regional_diet_model.pkl'),
                'diet_adherence_model': os.path.join(base_dir, 'Diet adherence Model', 'diet_adherence_model.pkl')
            }
            
            # Initialize engines
            engine = FoodRecommendationEngine(dataset_path, model_paths)
            gender_adjuster = GenderNutrientAdjustment()
            
            # 1. Run inference for base targets and metabolic class
            baseline, metabolic_class, adherence_score = engine.predict_user_profile(user_input)
            
            # 2. Get initial top 10 foods per meal (for re-ranking)
            initial_diet_plan = engine.generate_diet_plan(user_input, top_n=10)
            
            # 3. Apply Gender Nutrient adjustments (BMR + Re-ranking)
            final_plan = gender_adjuster.process(user_input, initial_diet_plan, meal_top_n=5)
            
            # --- Display Results ---
            st.success("✅ Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧬 Metabolic & Lifestyle Profile")
                st.info(f"**Metabolic Class:** {metabolic_class}")
                st.info(f"**Diet Adherence Score:** {adherence_score:.2f}")
                
            with col2:
                st.subheader("🎯 Daily Adjusted Targets (Mifflin-St Jeor)")
                st.metric("Daily Calories", f"{final_plan['adjusted_calorie_target']:.0f} kcal")
                
                macros = final_plan['macronutrient_targets']
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Protein", f"{macros.get('Protein (g)', 0):.0f}g")
                tc2.metric("Fat", f"{macros.get('Fat (g)', 0):.0f}g")
                tc3.metric("Carbs", f"{macros.get('Carbohydrates (g)', 0):.0f}g")
                
                micros = final_plan['micronutrient_targets']
                st.write("**Micronutrient Goals (Gender specific):**")
                for k, v in micros.items():
                    st.write(f"- {k}: {v}")
            
            st.divider()
            
            st.subheader("🍽️ Personalized Meal Plan (Gender Re-ranked)")
            st.markdown("Recommended foods dynamically adjusted based on Cosine Similarity to your customized baseline, then re-ranked by gender-specific micronutrient needs.")
            
            meal_data = final_plan['adjusted_meal_plan']
            
            # Create tabs for meals
            tabs = st.tabs(list(meal_data.keys()))
            
            for tab, (meal_name, items) in zip(tabs, meal_data.items()):
                with tab:
                    if not items:
                        st.write("No recommendations generated for this meal.")
                    else:
                        # Convert to DataFrame for nice table rendering
                        df_meal = pd.DataFrame(items)
                        cols_to_show = ['Food_Item', 'Category', 'Similarity Score', 'Calories (kcal)', 'Protein (g)', 'Fat (g)', 'Carbohydrates (g)']
                        existing_cols = [c for c in cols_to_show if c in df_meal.columns]
                        
                        st.dataframe(df_meal[existing_cols], use_container_width=True)
                        
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")
else:
    st.info("👈 Please set up your profile in the sidebar and click **Generate Recommendations** to get started.")
