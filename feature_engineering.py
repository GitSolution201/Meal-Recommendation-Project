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
        'Images', 'Description', 'DatePublished', 
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

def calculate_weight_loss_score(df):
    """
    Calculate a weight loss score based on nutritional features.
    Higher scores indicate better suitability for weight loss.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with nutritional features
    
    Returns:
    pandas.DataFrame: DataFrame with added WeightLossScore column
    """
    # Make a copy to avoid modifying the original
    df_scored = df.copy()
    
    # Create a copy of nutritional columns for scoring
    nutrition_cols = ['ProteinContent', 'FiberContent', 'FatContent', 'SugarContent', 'Calories']
    df_scored_for_scoring = df_scored[nutrition_cols].copy()
    
    # Normalize features to 0-1 scale for scoring only
    scaler = MinMaxScaler()
    df_scored_for_scoring = pd.DataFrame(
        scaler.fit_transform(df_scored_for_scoring),
        columns=nutrition_cols,
        index=df_scored.index
    )
    
    # Calculate Weight Loss Score with weighted components using normalized values
    df_scored['WeightLossScore'] = (
        0.5 * df_scored_for_scoring['ProteinContent'] +   # Highest priority (satiety)
        0.3 * df_scored_for_scoring['FiberContent'] -     # Fullness and digestion
        0.2 * df_scored_for_scoring['FatContent'] -       # Calorie density
        0.1 * df_scored_for_scoring['SugarContent'] -     # Blood sugar spikes
        0.05 * df_scored_for_scoring['Calories']          # Overall calorie control
    )
    
    # Normalize the final score to 0-1 range
    df_scored['WeightLossScore'] = (df_scored['WeightLossScore'] - df_scored['WeightLossScore'].min()) / \
                                  (df_scored['WeightLossScore'].max() - df_scored['WeightLossScore'].min())
    
    print("\nWeight Loss Score Summary:")
    print(f"Score Range: {df_scored['WeightLossScore'].min():.2f} to {df_scored['WeightLossScore'].max():.2f}")
    print(f"Mean Score: {df_scored['WeightLossScore'].mean():.2f}")
    
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

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('recipes.csv')
    df_selected = select_features_for_feature_Engineering(df)
    df_scored = calculate_weight_loss_score(df_selected)
    df_filtered = filter_meal_recipes(df_scored) 