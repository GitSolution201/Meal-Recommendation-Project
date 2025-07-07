import os
import csv
from datetime import datetime

def save_feedback_per_meal(user_profile, knn_recommendations, liked, feedback_file='user_feedback.csv'):
    """
    Save feedback for each meal in the plan, including user and meal info, meal number, and liked label.
    Args:
        user_profile (dict): User profile info (age, gender, goal, activity_level).
        knn_recommendations (pd.DataFrame): DataFrame with recommended meals.
        liked (bool): Feedback label for the meal plan.
        feedback_file (str): Path to the feedback CSV file.
    """
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'date',
                'user_age', 'user_gender', 'user_goal', 'activity_level',
                'meal_number',
                'meal_name', 'meal_calories', 'meal_protein',
                'meal_type_breakfast', 'meal_type_dinner',
                'is_vegetarian', 'has_high_sugar',
                'liked'
            ])
        for _, row in knn_recommendations.iterrows():
            meal_name = row.get('Name', '')
            meal_calories = row.get('Calories', '')
            meal_protein = row.get('ProteinContent', '')
            meal_type_breakfast = int('breakfast' in str(meal_name).lower() or row.get('MealType', '').lower() == 'breakfast')
            meal_type_dinner = int('dinner' in str(meal_name).lower() or row.get('MealType', '').lower() == 'dinner')
            is_vegetarian = int(row.get('is_vegetarian', 0))
            has_high_sugar = int(row.get('SugarContent', 0) > 15)
            meal_number = row.get('meal_number', '')
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d'),
                user_profile.get('age', ''),
                user_profile.get('gender', ''),
                user_profile.get('goal', ''),
                user_profile.get('activity_level', ''),
                meal_number,
                meal_name,
                meal_calories,
                meal_protein,
                meal_type_breakfast,
                meal_type_dinner,
                is_vegetarian,
                has_high_sugar,
                int(liked)
            ])

def save_meal_plan_feedback_detailed(user_id, meal_names, meal_calories, liked, feedback_file='user_feedback_detailed.csv'):
    """
    Save user feedback for confusion matrix: user_id, meal_1, meal_2, meal_3, calories_1, calories_2, calories_3, liked.
    Args:
        user_id (str): Identifier for the user.
        meal_names (list of str): Names of the 3 meals.
        meal_calories (list of float/int): Calories for the 3 meals.
        liked (bool): True if user likes the meal plan, False otherwise.
        feedback_file (str): Path to the feedback CSV file.
    """
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'user_id',
                'meal_1', 'meal_2', 'meal_3',
                'calories_1', 'calories_2', 'calories_3',
                'liked'
            ])
        row = [user_id]
        row += meal_names[:3] + [''] * (3 - len(meal_names))  # pad if less than 3
        row += [str(c) for c in meal_calories[:3]] + [''] * (3 - len(meal_calories))
        row.append(int(liked))
        writer.writerow(row)

def save_detailed_feedback_per_meal(user_profile, knn_recommendations, liked, feedback_file='user_feedback_full.csv'):
    """
    Save detailed feedback for each meal in the plan, including user and meal info, and feedback label.
    Args:
        user_profile (dict): User profile info (age, gender, goal, activity_level).
        knn_recommendations (pd.DataFrame): DataFrame with recommended meals.
        liked (bool): Feedback label for the meal plan.
        feedback_file (str): Path to the feedback CSV file.
    """
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'date',
                'user_age', 'user_gender', 'user_goal', 'activity_level',
                'meal_name', 'meal_calories', 'meal_protein',
                'meal_type_breakfast', 'meal_type_dinner',
                'is_vegetarian', 'has_high_sugar',
                'feedback'
            ])
        for _, row in knn_recommendations.head(3).iterrows():
            meal_name = row.get('Name', '')
            meal_calories = row.get('Calories', '')
            meal_protein = row.get('ProteinContent', '')
            # Binary flags for meal type (assume columns or infer from name/type)
            meal_type_breakfast = int('breakfast' in str(meal_name).lower() or row.get('MealType', '').lower() == 'breakfast')
            meal_type_dinner = int('dinner' in str(meal_name).lower() or row.get('MealType', '').lower() == 'dinner')
            is_vegetarian = int(row.get('is_vegetarian', 0))
            has_high_sugar = int(row.get('SugarContent', 0) > 15)  # threshold can be adjusted
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d'),
                user_profile.get('age', ''),
                user_profile.get('gender', ''),
                user_profile.get('goal', ''),
                user_profile.get('activity_level', ''),
                meal_name,
                meal_calories,
                meal_protein,
                meal_type_breakfast,
                meal_type_dinner,
                is_vegetarian,
                has_high_sugar,
                int(liked)
            ])

def collect_and_save_feedback(knn_recommendations, user_profile=None):
    """
    Handles user feedback interaction, collects user_id, extracts meal names/calories, and saves feedback.
    Args:
        knn_recommendations (pd.DataFrame): DataFrame with recommended meals (must have 'Name' and 'Calories' columns)
        user_profile (dict, optional): User profile info (age, gender, goal, activity_level)
    """
    response = input("Do you like the recommended meal plan? (y/n): ").strip().lower()
    user_id = input("Enter your user ID (or leave blank): ").strip()
    top_meals = list(knn_recommendations['Name'].head(3))
    top_calories = list(knn_recommendations['Calories'].head(3))
    liked = response == 'y'
    if user_profile is None:
        user_profile = {}
    # Save old feedback formats for compatibility
    save_meal_plan_feedback(top_meals, liked=liked, user_id=user_id)
    save_meal_plan_feedback_detailed(user_id, top_meals, top_calories, liked=liked)
    # Save new detailed feedback per meal
    save_detailed_feedback_per_meal(user_profile, knn_recommendations, liked)
    if liked:
        print("Thank you for your feedback! Meal plan saved to user_feedback.csv, user_feedback_detailed.csv, and user_feedback_full.csv.")
    else:
        print("Sorry these meals didn't suit you. We will suggest a new meal plan! Your feedback was saved for analysis.") 