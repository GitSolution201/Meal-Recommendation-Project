import pandas as pd

def recommend_meals_for_user(df, user_profile):
    """
    Recommend meals based on user's calorie goals and nutritional needs.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with recipe data and weight loss scores
    user_profile (dict): User profile dictionary with calorie goals and preferences
    
    Returns:
    pandas.DataFrame: Recommended meals with nutritional information
    """
    # Make a copy to avoid modifying the original
    df_recommended = df.copy()
    
    # Calculate daily meal calorie targets (assuming 3 meals + 2 snacks)
    meal_calorie_target = user_profile['CalorieGoal'] * 0.3  # 30% for main meals
    # snack_calorie_target = user_profile['CalorieGoal'] * 0.1  # 10% for snacks
    
    # Filter recipes based on calorie content
    df_recommended = df_recommended[
        (df_recommended['Calories'] <= meal_calorie_target) &
        (df_recommended['WeightLossScore'] > 0.5)  # Only high-scoring recipes
    ]
    
    # Sort by weight loss score and calories
    df_recommended = df_recommended.sort_values(
        ['WeightLossScore', 'Calories'],
        ascending=[False, True]
    )
    
    # Select top recommendations
    recommendations = df_recommended.head(5)
    
    print("\n=== Meal Recommendations ===")
    print(f"Based on your daily calorie goal of {user_profile['CalorieGoal']} calories")
    print(f"Target calories per meal: {meal_calorie_target:.0f}")
    print("\nRecommended Meals:")
    
    for idx, meal in recommendations.iterrows():
        print(f"\n{meal['Name']}")
        print(f"Calories: {meal['Calories']:.0f}")
        print(f"Protein: {meal['ProteinContent']:.1f}g")
        print(f"Fiber: {meal['FiberContent']:.1f}g")
        print(f"Weight Loss Score: {meal['WeightLossScore']:.2f}")
    
    return recommendations

def create_meal_plan(recommendations, user_profile):
    """
    Create a weekly meal plan based on recommendations.
    
    Parameters:
    recommendations (pandas.DataFrame): Recommended meals
    user_profile (dict): User profile dictionary
    
    Returns:
    dict: Weekly meal plan
    """
    # Calculate daily calorie distribution
    daily_calories = user_profile['CalorieGoal']
    meal_distribution = {
        'Breakfast': 0.25,  # 25% of daily calories
        'Lunch': 0.35,      # 35% of daily calories
        'Dinner': 0.40      # 40% of daily calories
    }
    
    # Create meal plan
    meal_plan = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        meal_plan[day] = {
            'Breakfast': recommendations.sample(1).iloc[0].to_dict(),
            'Lunch': recommendations.sample(1).iloc[0].to_dict(),
            'Dinner': recommendations.sample(1).iloc[0].to_dict()
        }
    
    return meal_plan

def print_meal_plan(meal_plan):
    """
    Print the weekly meal plan in a readable format.
    
    Parameters:
    meal_plan (dict): Weekly meal plan
    """
    print("\n=== Weekly Meal Plan ===")
    for day, meals in meal_plan.items():
        print(f"\n{day}:")
        for meal_type, meal in meals.items():
            print(f"\n{meal_type}:")
            print(f"- {meal['Name']}")
            print(f"  Calories: {meal['Calories']:.0f}")
            print(f"  Protein: {meal['ProteinContent']:.1f}g")
            print(f"  Fiber: {meal['FiberContent']:.1f}g")

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from user_profile import get_user_profile
    
    # Create sample data
    df = pd.read_csv('recipes.csv')
    user_profile = get_user_profile(
        age=20,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    
    # Get recommendations and create meal plan
    recommendations = recommend_meals_for_user(df, user_profile)
    meal_plan = create_meal_plan(recommendations, user_profile)
    print_meal_plan(meal_plan) 