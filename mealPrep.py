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
import os
from feedback import save_feedback_per_meal

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
    # EDA_graphs(df_cleaned)

    # Check data quality
    # quality_metrics = check_data_quality(df_cleaned)
    
    # Select features for feature engineering
    df_selected = select_features_for_feature_Engineering(df_cleaned)
    # Calculate weight loss score
    df_with_scores = calculate_weight_loss_score(df_selected)
    print("weight loss scores------",df_with_scores)
    # Filter meal recipes
    df_filtered = filter_meal_recipes(df_with_scores)

    # Print and save top 10 entries of df_filtered to CSV for inspection
    print("\nTop 10 rows of data passed to KNN (df_filtered):")
    print(df_filtered.head(10))
    df_filtered.head(40).to_csv('df_filtered_top10.csv', index=False)
    from sklearn.model_selection import train_test_split
    # Save all of df_filtered to CSV for use in training
    # Split into train and test files
    train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    train_df.to_csv('filtered_meals_train.csv', index=False)
    test_df.to_csv('filtered_meals_test.csv', index=False)
    print("Train and test files created!")
    
    # Example user profile (static values)
    age = 89
    gender = "female"
    weight_kg = 180
    height_cm = 175
    activity_level = "light"
    goal = "moderate"
    user_profile = get_user_profile(
        age=age,
        gender=gender,
        weight_kg=weight_kg,
        height_cm=height_cm,
        activity_level=activity_level,
        goal=goal
    )
    print("---------------------profile",user_profile)
    # Recommend meals for user
    # recommendations = recommend_meals_for_user(df_filtered, user_profile)
    print("\nRecommended Meals DataFrame:")
    
    # --- KNN-based Recommendation ---
    knn_features = [
    'Calories',
    'FatContent',
    'SaturatedFatContent',
    'CholesterolContent',
    'SodiumContent',
    'CarbohydrateContent',
    'FiberContent',
    'SugarContent',
    'ProteinContent'
]
    # inputing features that defines the similarity / distance between meals 
    
    knn_recommendations = knn_recommend_meals(df_filtered, user_profile, knn_features, n_neighbors=3)
    print(knn_recommendations[['Name'] + knn_features])

    # --- Precision and Recall for KNN ---
    # Simulate relevant meal IDs (for demo, pick 3 from recommendations)
    relevant_meal_ids = set(knn_recommendations['RecipeId'].head(5)) if 'RecipeId' in knn_recommendations.columns else set(knn_recommendations.index[:10])
    #relevant_meal_ids are those meals user prefered 
    print(relevant_meal_ids)
    k = 10

    # so precision is taking the todal number of recommendations plus user liked recommendations and then k is evalution of restults from top to botom
    precision = precision_at_k_knn(knn_recommendations, relevant_meal_ids, k)
    
    recall = recall_at_k_knn(knn_recommendations, relevant_meal_ids, k)
    print(f"\nPrecision@{k} for KNN: {precision:.2f}")
    print(f"Recall@{k} for KNN: {recall:.2f}")

    # --- User Feedback Section (per meal, with BMI and BMR) ---
    bmi = user_profile['BMI']
    bmr = user_profile['BMR']
    for idx, meal_row in enumerate(knn_recommendations.head(3).iterrows(), 1):
        _, meal_row_data = meal_row
        meal_name = meal_row_data.get('Name', '')
        response = input(f"Did you like the meal '{meal_name}'? (y/n): ").strip().lower()
        liked = response == 'y'
        # Add meal_number to the row if needed
        meal_row_data = meal_row_data.copy()
        single_meal_df = pd.DataFrame([meal_row_data])
        save_feedback_per_meal(age, gender, activity_level, goal, bmi, bmr, single_meal_df, liked)
    print("Your feedback for each meal has been saved to user_feedback.csv.")

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield:
print("\nUpdated Column Names after dropping RecipeYield:")

