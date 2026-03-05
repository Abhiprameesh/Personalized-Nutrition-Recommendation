class GenderNutrientAdjustment:
    """
    Module to adjust nutritional targets and recommended foods based on gender-specific dietary requirements.
    """
    
    def __init__(self):
        pass

    def calculate_bmr(self, gender, age, weight, height):
        """
        Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation.
        """
        gender = gender.lower()
        if gender == 'female':
            return 10 * weight + 6.25 * height - 5 * age - 161
        else:
            return 10 * weight + 6.25 * height - 5 * age + 5

    def estimate_daily_calories(self, bmr, activity_level):
        """
        Estimate daily calorie needs based on BMR and physical activity level.
        activity_level mapping: 1 (Sedentary) to 4 (Very Active)
        """
        activity_multiplier = {
            1: 1.2,     # Sedentary
            2: 1.375,   # Lightly active
            3: 1.55,    # Moderately active
            4: 1.725    # Very active
        }.get(activity_level, 1.375) # Default to lightly active
        
        return bmr * activity_multiplier

    def get_macronutrient_targets(self, calories):
        """
        Convert calorie targets into macronutrient targets using a typical distribution:
        Protein: 25%, Fat: 25%, Carbohydrates: 50%
        """
        return {
            'Protein (g)': (calories * 0.25) / 4,
            'Fat (g)': (calories * 0.25) / 9,
            'Carbohydrates (g)': (calories * 0.50) / 4
        }

    def get_micronutrient_targets(self, gender):
        """
        Adjust micronutrient targets based on gender using clinical guidelines.
        """
        gender = gender.lower()
        if gender == 'female':
            return {
                'Iron (mg)': 18,
                'Calcium (mg)': 1200,
                'Fiber (g)': 25
            }
        else:
            return {
                'Iron (mg)': 8,
                'Calcium (mg)': 1000,
                'Fiber (g)': 30
            }

    def modify_recommended_foods(self, initial_recommendations, gender, top_n=3):
        """
        Modify recommended foods from the food recommendation engine based on gender.
        Females: prioritize iron-rich foods (spinach, lentils, beans).
        Males: prioritize higher protein foods (chicken, eggs, fish).
        """
        gender = gender.lower()
        adjusted_plan = {}
        
        female_iron_keywords = ['spinach', 'lentil', 'bean', 'kale', 'beef', 'chickpea']
        male_protein_keywords = ['chicken', 'egg', 'fish', 'salmon', 'tuna', 'beef', 'steak', 'pork']

        for meal, foods in initial_recommendations.items():
            sorted_foods = list(foods)
            
            if gender == 'female':
                # Prioritize iron-rich keywords
                sorted_foods.sort(
                    key=lambda x: any(kw in x['Food_Item'].lower() for kw in female_iron_keywords), 
                    reverse=True
                )
            else:
                # Prioritize high protein keywords, followed by actual protein content
                sorted_foods.sort(
                    key=lambda x: (
                        any(kw in x['Food_Item'].lower() for kw in male_protein_keywords),
                        x.get('Protein (g)', 0)
                    ), 
                    reverse=True
                )
                
            adjusted_plan[meal] = sorted_foods[:top_n]
            
        return adjusted_plan

    def process(self, user_profile, initial_recommendations, meal_top_n=3):
        """
        Main pipeline method to process user profile and recommendations.
        """
        gender = user_profile.get('Gender', 'Male')
        age = user_profile.get('Age', 30)
        weight = user_profile.get('Weight', 70)
        height = user_profile.get('Height', 170)
        activity_level = user_profile.get('Physical Activity Level', 2)

        # 1. Calculate BMR
        bmr = self.calculate_bmr(gender, age, weight, height)
        
        # 2. Estimate Calories
        adjusted_calories = self.estimate_daily_calories(bmr, activity_level)
        
        # 3. Macro & Micro targets
        macros = self.get_macronutrient_targets(adjusted_calories)
        micros = self.get_micronutrient_targets(gender)
        
        # 4. Modify food recommendations
        adjusted_plan = self.modify_recommended_foods(initial_recommendations, gender, top_n=meal_top_n)
        
        return {
             'adjusted_calorie_target': round(adjusted_calories, 2),
             'macronutrient_targets': {k: round(v, 2) for k, v in macros.items()},
             'micronutrient_targets': micros,
             'adjusted_meal_plan': adjusted_plan
        }
