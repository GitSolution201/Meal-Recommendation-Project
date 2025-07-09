import os
import csv
from datetime import datetime

def save_feedback_per_meal(age, gender, activity_level, goal, bmi, bmr, knn_recommendations, liked, feedback_file='user_feedback.csv'):
    """
    Save feedback for each meal in the plan, including user and meal info, BMI, BMR, weight loss score, and liked label.
    Args:
        age (int/str): User's age.
        gender (str): User's gender.
        activity_level (str): User's activity level.
        goal (str): User's goal.
        bmi (float): User's BMI.
        bmr (float): User's BMR.
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
                'age', 'gender', 'activity_level', 'goal',
                'BMI', 'BMR', 'weight_loss_score',
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
            weight_loss_score = row.get('weight_loss_score', '')
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d'),
                age,
                gender,
                activity_level,
                goal,
                bmi,
                bmr,
                weight_loss_score,
                meal_name,
                meal_calories,
                meal_protein,
                meal_type_breakfast,
                meal_type_dinner,
                is_vegetarian,
                has_high_sugar,
                int(liked)
            ]) 