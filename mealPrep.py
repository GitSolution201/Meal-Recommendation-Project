import pandas as pd
from data_cleaning import perform_EDA, check_data_quality
from feature_engineering import (
    select_features_for_feature_Engineering,
    calculate_weight_loss_score,
    filter_meal_recipes
)
from meal_recommendations import show_best_worst_meals

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
    
    # Show best and worst meals
    show_best_worst_meals(df_filtered)

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

