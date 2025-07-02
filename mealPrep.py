import pandas as pd
from data_cleaning import perform_EDA, check_data_quality
from feature_engineering import (
    select_features_for_feature_Engineering,
    calculate_weight_loss_score,
    filter_meal_recipes
)
from meal_recommendations import show_best_worst_meals, recommend_meals_for_user, precision_at_k_knn, recall_at_k_knn
from user_profile import get_user_profile
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda_graphs import EDA_graphs
from knn_recommender import knn_recommend_meals

def main():
    # Read the CSV file
    df = pd.read_csv('recipes.csv')
    print(f"Total number of rows in the dataset: {len(df)}")

    # Find and print food(s) with the highest and lowest calories
    max_cal = df['Calories'].max()
    min_cal = df['Calories'].min()
    print("\nFood(s) with the highest calories:")
    print(df[df['Calories'] == max_cal][['Name', 'Calories']])
    print("\nFood(s) with the lowest calories:")
    print(df[df['Calories'] == min_cal][['Name', 'Calories']])
    
    # Perform EDA and cleaning
    df_cleaned = perform_EDA(df)
    print(df_cleaned.columns)

    # Call EDA graphs function
    EDA_graphs(df_cleaned)

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
    knn_features = ['Calories', 'ProteinContent', 'FatContent', 'FiberContent', 'WeightLossScore']
    knn_recommendations = knn_recommend_meals(df_filtered, user_profile, knn_features, n_neighbors=5)
    print("\nKNN-based Recommended Meals:")
    print(knn_recommendations[['Name'] + knn_features])

    # --- Precision and Recall for KNN ---
    # Simulate relevant meal IDs (for demo, pick 3 from recommendations)
    relevant_meal_ids = set(knn_recommendations['RecipeId'].head(3)) if 'RecipeId' in knn_recommendations.columns else set(knn_recommendations.index[:3])
    k = 5
    precision = precision_at_k_knn(knn_recommendations, relevant_meal_ids, k)
    recall = recall_at_k_knn(knn_recommendations, relevant_meal_ids, k)
    print(f"\nPrecision@{k} for KNN: {precision:.2f}")
    print(f"Recall@{k} for KNN: {recall:.2f}")

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

