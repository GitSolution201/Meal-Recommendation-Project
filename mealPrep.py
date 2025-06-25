import pandas as pd
from data_cleaning import perform_EDA, check_data_quality
from feature_engineering import (
    select_features_for_feature_Engineering,
    calculate_weight_loss_score,
    filter_meal_recipes
)
from meal_recommendations import show_best_worst_meals, recommend_meals_for_user
from user_profile import get_user_profile

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
    
    # Example user profile (replace with dynamic/user input as needed)
    user_profile = get_user_profile(
        age=30,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    
    # Recommend meals for user
    recommendations = recommend_meals_for_user(df_filtered, user_profile)
    print("\nRecommended Meals DataFrame:")
    print(recommendations)

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

