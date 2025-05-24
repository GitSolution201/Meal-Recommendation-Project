import pandas as pd
from data_cleaning import perform_EDA, check_data_quality
from feature_engineering import (
    select_features_for_feature_Engineering,
    calculate_weight_loss_score,
    filter_meal_recipes
)
from user_profile import get_user_profile, print_user_profile
from meal_recommendations import (
    recommend_meals_for_user,
    create_meal_plan,
    print_meal_plan
)

def main():
    # Read the CSV file
    df = pd.read_csv('recipes.csv')
    
    # Perform EDA and cleaning
    df_cleaned = perform_EDA(df)
    
    # Check data quality
    quality_metrics = check_data_quality(df_cleaned)
    
    # Select features for feature engineering
    df_selected = select_features_for_feature_Engineering(df_cleaned)
    
    # Calculate weight loss score
    df_with_scores = calculate_weight_loss_score(df_selected)
    
    # Filter meal recipes
    df_filtered = filter_meal_recipes(df_with_scores)
    
    # Create user profile
    user_profile = get_user_profile(
        age=20,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    
    # Print user profile
    print_user_profile(user_profile)
    
    # Get meal recommendations
    recommendations = recommend_meals_for_user(df_filtered, user_profile)
    
    # Create and print weekly meal plan
    meal_plan = create_meal_plan(recommendations, user_profile)
    print_meal_plan(meal_plan)

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

