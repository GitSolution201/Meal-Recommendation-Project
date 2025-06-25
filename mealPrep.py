import pandas as pd
from data_cleaning import perform_EDA, check_data_quality
from feature_engineering import (
    select_features_for_feature_Engineering,
    calculate_weight_loss_score,
    filter_meal_recipes
)
from meal_recommendations import show_best_worst_meals, recommend_meals_for_user
from user_profile import get_user_profile
from sklearn.neighbors import NearestNeighbors
import numpy as np

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

    # --- KNN-based Recommendation ---
    # Define the features to use for KNN
    knn_features = ['Calories', 'ProteinContent', 'FatContent', 'FiberContent', 'WeightLossScore']
    #You define the numerical features that describe each meal.
    X = df_filtered[knn_features]
    
    # Create a user profile vector based on calorie goal and healthy targets (example values)
    #You create a "target meal profile" for the user — this is what the user is aiming for in each meal.

    #user_vector is startig point
    user_vector = np.array([
        user_profile['CalorieGoal'] * 0.3,  # per meal
        75,  # target protein per meal (example)
        10,  # target fat per meal (example)
        100,   # target fiber per meal (example)
        1.0  # ideal weight loss score
    ]).reshape(1, -1)
    #Reshape it to a 2D array because kneighbors() expects that format.


    
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=5, metric='manhattan')
    knn.fit(X)
    distances, indices = knn.kneighbors(user_vector)
    knn_recommendations = df_filtered.iloc[indices[0]]
    print("\nKNN-based Recommended Meals:")
    print(knn_recommendations[['Name'] + knn_features])

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

