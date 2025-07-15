import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def select_features_for_feature_Engineering(df):
    """
    Select features for feature engineering by dropping irrelevant columns.
    
    Parameters:
    df (pandas.DataFrame): The cleaned DataFrame from EDA
    
    Returns:
    pandas.DataFrame: DataFrame with selected features for feature engineering
    """
    # Make a copy to avoid modifying the original
    df_selected = df.copy()
    
    # List of columns to drop
    columns_to_drop = [
        'RecipeId', 'AuthorId', 'AuthorName', 
        # 'Images',  # Do not drop Images column
        'Description', 'DatePublished', 
        'RecipeYield', 'RecipeIngredientQuantities', 
        'ReviewCount', 'AggregatedRating'
    ]
    
    # Drop specified columns
    df_selected = df_selected.drop(columns=columns_to_drop, errors='ignore')
    
    # Print information about the selection
    print("\nFeature Engineering Selection Summary:")
    print(f"Original number of columns: {len(df.columns)}")
    print(f"Number of columns after selection: {len(df_selected.columns)}")
    print("\nSelected columns:")
    for col in df_selected.columns:
        print(f"- {col}")
    
    return df_selected
#You are calculating a Weight Loss Score to quantitatively evaluate and compare how suitable each recipe is for weight loss, based on its nutritional content.

def calculate_weight_loss_score(df, user_profile=None):
    """
    Calculate a weight loss score based on nutritional features only.
    Higher scores indicate better suitability for weight loss.
    Parameters:
    df (pandas.DataFrame): DataFrame with nutritional features
    user_profile (dict): (Unused here, only for personalized rules)
    Returns:
    pandas.DataFrame: DataFrame with added WeightLossScore column
    """
    # Make a copy to avoid modifying the original
    df_scored = df.copy()
    # Create a copy of nutritional columns for scoring
    nutrition_cols = [
        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
    ]
    df_scored_for_scoring = df_scored[nutrition_cols].copy()
    # Normalize features to 0-1 scale for scoring only
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scored_for_scoring = pd.DataFrame(
        scaler.fit_transform(df_scored_for_scoring),
        columns=nutrition_cols,
        index=df_scored.index
    )
    # Calculate base Weight Loss Score with weighted components using normalized values
    base_score = (
        10 * df_scored_for_scoring['ProteinContent'] +      # High priority (satiety)
        5 * df_scored_for_scoring['FiberContent'] -        # Fullness and digestion
        5 * df_scored_for_scoring['FatContent'] -          # Calorie density
        5 * df_scored_for_scoring['SugarContent'] -        # Blood sugar spikes
        5 * df_scored_for_scoring['Calories'] -            # Overall calorie control
        5 * df_scored_for_scoring['SaturatedFatContent'] - # Heart health
        5 * df_scored_for_scoring['CholesterolContent'] +  # Heart health
        2 * df_scored_for_scoring['SodiumContent'] -       # Blood pressure
        5 * df_scored_for_scoring['CarbohydrateContent']   # Energy balance
    )
    # Only use base score (no user profile adjustment)
    df_scored['WeightLossScore'] = base_score
    return df_scored

def filter_meal_recipes(df):
    """
    Filter out non-meal recipes based on their categories and names.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with recipe data
    
    Returns:
    pandas.DataFrame: DataFrame containing only meal recipes
    """
    # Make a copy to avoid modifying the original
    df_filtered = df.copy()
    
    # Define non-meal categories to exclude
    non_meals = ['Sauce', 'Beverage', 'Dessert', 'Drink', 'Snack', 'Appetizer', 'Condiment']
    
    # Filter out non-meal categories
    df_filtered = df_filtered[~df_filtered['RecipeCategory'].isin(non_meals)]
    
    # Additional filtering based on recipe names
    non_meal_keywords = ['sauce', 'drink', 'dessert', 'beverage', 'moose', 'whiskey', 'wine', 'beer']
    df_filtered = df_filtered[~df_filtered['Name'].str.lower().str.contains('|'.join(non_meal_keywords), na=False)]
    
    print("\nRecipe Filtering Summary:")
    print(f"Original number of recipes: {len(df)}")
    print(f"Number of recipes after filtering: {len(df_filtered)}")
    print(f"Removed {len(df) - len(df_filtered)} non-meal recipes")
    
    # Display remaining categories
    print("\nRemaining recipe categories:")
    print(df_filtered['RecipeCategory'].value_counts())
    
    return df_filtered

def show_weight_loss_score_distribution(df):
    """
    Print and plot the distribution, min, and max of the WeightLossScore column.
    Args:
        df (pd.DataFrame): DataFrame with WeightLossScore column
    """
    import matplotlib.pyplot as plt
    print("\nWeightLossScore Distribution:")
    print(df['WeightLossScore'].describe())
    min_val = df['WeightLossScore'].min()
    max_val = df['WeightLossScore'].max()
    mean_val = df['WeightLossScore'].mean()
    print(f"Min: {min_val:.2f}")
    print(f"Max: {max_val:.2f}")
    print(f"Mean: {mean_val:.2f}")
    plt.figure(figsize=(8, 5))
    plt.hist(df['WeightLossScore'], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('WeightLossScore')
    plt.ylabel('Number of Meals')
    plt.title('Distribution of WeightLossScore')
    plt.grid(axis='y', alpha=0.75)
    # Highlight min and max values
    plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(max_val, color='green', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')
    plt.legend()
    plt.show()

def random_user_profile():
    age = random.randint(15, 40)
    weight_kg = random.randint(40, 150)
    gender = random.choice(['male', 'female'])
    height_cm = random.randint(150, 200)
    # Simple BMI calculation
    bmi = weight_kg / ((height_cm / 100) ** 2)
    # Simple BMR calculation (Mifflin-St Jeor Equation)
    if gender == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    calorie_goal = int(bmr * random.uniform(1.2, 1.5))
    return {
        'BMI': bmi,
        'BMR': bmr,
        'age': age,
        'weight_kg': weight_kg,
        'gender': gender,
        'CalorieGoal': calorie_goal
    }

def is_good_meal(meal, user_profile):
    meal_target_calories = user_profile['BMR'] * 1 / 3
    protein_target = user_profile['weight_kg'] * 0.2 / 3  # Example threshold
    fiber_threshold = 3  # Example threshold
    wls_threshold = 0.5  # Example threshold

    return (
        abs(meal['Calories'] - meal_target_calories) < 0.5 * meal_target_calories or
        meal['ProteinContent'] >= protein_target and
        meal['FiberContent'] >= fiber_threshold and
        meal['WeightLossScore'] > wls_threshold
    )

def classify_meal_goodness_by_percentile_random_users(df):
    """
    For every 5 meals, generate a new random user profile and classify meals accordingly.
    Also save the user profile (age, weight_kg, gender, BMI, BMR) for each meal.
    """
    df = df.copy()
    user_profiles = [random_user_profile() for _ in range((len(df) // 5) + 1)]
    is_good_list = []
    ages, weights, genders, bmis, bmrs = [], [], [], [], []
    for i, (_, meal) in enumerate(df.iterrows()):
        user_profile = user_profiles[i // 5]
        is_good = is_good_meal(meal, user_profile)
        is_good_list.append(int(is_good))
        ages.append(user_profile['age'])
        weights.append(user_profile['weight_kg'])
        genders.append(user_profile['gender'])
        bmis.append(user_profile['BMI'])
        bmrs.append(user_profile['BMR'])
    df['IsGoodMeal'] = is_good_list
    df['UserAge'] = ages
    df['UserWeight'] = weights
    df['UserGender'] = genders
    df['UserBMI'] = bmis
    df['UserBMR'] = bmrs
    return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('recipes.csv')
    df_selected = select_features_for_feature_Engineering(df)
    
    # Example user profile for personalized scoring
    example_user_profile = {
        'BMI': 28.5,
        'BMR': 1650,
        'age': 35,
        'weight_kg': 80,
        'CalorieGoal': 1800
    }
    
    # Calculate weight loss score with user profile
    df_scored = calculate_weight_loss_score(df_selected, user_profile=example_user_profile)
    show_weight_loss_score_distribution(df_scored)
    df_classified = classify_meal_goodness_by_percentile_random_users(df_scored)
    print("--------------------",df_classified[['WeightLossScore', 'IsGoodMeal']].head())
    df_classified.head(40).to_csv('classified_meals.csv', index=False)
    # Print the number of good and non-good meals
    print('Number of good meals:', (df_classified['IsGoodMeal'] == 1).sum())
    print('Number of non-good meals:', (df_classified['IsGoodMeal'] == 0).sum())
    df_filtered = filter_meal_recipes(df_classified) 