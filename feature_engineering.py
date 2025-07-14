import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

def classify_meal_goodness_by_percentile(df, user_profile=None):
    """
    Assign IsGoodMeal randomly for testing model performance with random labels.
    Args:
        df (pd.DataFrame): DataFrame with WeightLossScore and nutritional columns
        user_profile (dict): (Unused)
    Returns:
        pd.DataFrame: DataFrame with IsGoodMeal column
    """
    import numpy as np
    df = df.copy()
    df['IsGoodMeal'] = np.random.randint(0, 2, size=len(df))
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
    df_classified = classify_meal_goodness_by_percentile(df_scored, user_profile=example_user_profile)
    print("--------------------",df_classified[['WeightLossScore', 'IsGoodMeal']].head())
    df_classified.to_csv('classified_meals.csv', index=False)
    # Print the number of good and non-good meals
    print('Number of good meals:', (df_classified['IsGoodMeal'] == 1).sum())
    print('Number of non-good meals:', (df_classified['IsGoodMeal'] == 0).sum())
    df_filtered = filter_meal_recipes(df_classified) 