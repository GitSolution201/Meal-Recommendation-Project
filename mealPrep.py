import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set display options to show all columns and full content
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

def perform_EDA(df):
    """
    Perform Exploratory Data Analysis and data cleaning on the recipes dataset.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing recipe data
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()
    
    # Check for duplicates
    print("\nChecking for duplicates...")
    duplicates = df_clean.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")
    
    # Remove duplicates if any exist
    if duplicates.sum() > 0:
        print("Removing duplicate rows...")
        df_clean = df_clean.drop_duplicates()
        print(f"Rows after removing duplicates: {len(df_clean)}")
    
    # 1. Drop RecipeYield column as it has low utility
    df_clean = df_clean.drop('RecipeYield', axis=1)
    
    # 2. Handle AggregatedRating - fill missing values with average rating
    avg_rating = df_clean['AggregatedRating'].mean()
    df_clean['AggregatedRating'] = df_clean['AggregatedRating'].fillna(avg_rating)
    
    # 3. Handle ReviewCount - fill missing values with 0 (no reviews)
    df_clean['ReviewCount'] = df_clean['ReviewCount'].fillna(0)
    
    # 4. Handle RecipeServings - fill missing values with median
    median_servings = df_clean['RecipeServings'].median()
    df_clean['RecipeServings'] = df_clean['RecipeServings'].fillna(median_servings)
    
    # 5. Handle CookTime - if TotalTime exists, we can drop CookTime
    # First, let's check if we have TotalTime
    if 'TotalTime' in df_clean.columns:
        df_clean = df_clean.drop('CookTime', axis=1)
    
    # 6. Handle Keywords - fill missing values with empty string
    df_clean['Keywords'] = df_clean['Keywords'].fillna("")
    
    # 7. Handle RecipeCategory - fill missing values with "Unknown"
    df_clean['RecipeCategory'] = df_clean['RecipeCategory'].fillna("Unknown")
    
    # 8. Handle Description - fill missing values with empty string
    df_clean['Description'] = df_clean['Description'].fillna("")
    
    # 9. Handle RecipeIngredientQuantities - fill missing values with empty string
    # This is a careful approach as we don't want to make assumptions about quantities
    df_clean['RecipeIngredientQuantities'] = df_clean['RecipeIngredientQuantities'].fillna("")
    
    # 10. Handle Images - fill missing values with empty string
    df_clean['Images'] = df_clean['Images'].fillna("")
    
    # Print summary of changes
    print("\nData Cleaning Summary:")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_clean.shape}")
    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum().sort_values(ascending=False))
    
    return df_clean

def select_features_for_heatmap(df):
    """
    Select relevant features for analysis and drop irrelevant ones.
    
    Parameters:
    df (pandas.DataFrame): The cleaned DataFrame from EDA
    
    Returns:
    pandas.DataFrame: DataFrame with selected features
    """
    # List of columns to keep (nutritional features)
    nutritional_features = [
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
    
    # List of columns to drop
    columns_to_drop = [
        'RecipeId', 'AuthorId', 'AuthorName',  # irrelevant identifiers
        'Name', 'Description', 'Images', 'RecipeInstructions',  # unstructured text
        'ReviewCount', 'AggregatedRating',  # popularity metrics
        'DatePublished',  # not relevant
        'RecipeIngredientQuantities', 'RecipeIngredientParts',  # complex features
        'RecipeYield',
        'PrepTime',
        'TotalTime',
        'Keywords',
        'RecipeServings',
        'RecipeCategory'
          # already dropped in EDA
    ]
    
    # Make a copy of the DataFrame
    df_selected = df.copy()
    
    # Drop specified columns
    df_selected = df_selected.drop(columns=columns_to_drop, errors='ignore')
    
    # Print information about the selection
    print("\nFeature Selection Summary:")
    print(f"Original number of columns: {len(df.columns)}")
    print(f"Number of columns after selection: {len(df_selected.columns)}")
    print("\nSelected columns:")
    for col in df_selected.columns:
        print(f"- {col}")
    
    return df_selected

def create_correlation_heatmap(df):
    """
    Create a heatmap showing correlations between nutritional features.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with selected nutritional features
    """
    print("\nData types of df_selected columns:")
    print(df.dtypes)
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create figure with a larger size
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True,  # Show correlation values
                cmap='coolwarm',  # Color scheme
                center=0,  # Center the colormap at 0
                fmt='.2f',  # Format correlation values to 2 decimal places
                square=True)  # Make the plot square-shaped
    
    # Add title
    plt.title('Correlation Heatmap of Nutritional Features', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('nutritional_correlation_heatmap.png')
    print("\nCorrelation heatmap has been saved as 'nutritional_correlation_heatmap.png'")
    
    # Show the plot
    plt.show()

def select_features_for_feature_Engineering(df):
    """
    Select features for feature engineering by dropping irrelevant columns.
    
    Parameters:
    df (pandas.DataFrame): The cleaned DataFrame from EDA
    
    Returns:
    pandas.DataFrame: DataFrame with selected features for feature engineering
    """
    # Make a copy of the DataFrame to avoid modifying the original
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
    # for col in df_selected.columns:
    #     print(f"- {col}")
    
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
    
    # Select nutritional columns for scoring
    nutrition_cols = ['ProteinContent', 'FiberContent', 'FatContent', 'SugarContent', 'Calories']
    
    # Normalize features to 0-1 scale
    scaler = MinMaxScaler()
    df_scored[nutrition_cols] = scaler.fit_transform(df_scored[nutrition_cols])
    
    # Calculate Weight Loss Score with corrected weights
    df_scored['WeightLossScore'] = (
        0.5 * df_scored['ProteinContent'] +   # Prioritize protein (satiety)
        0.3 * df_scored['FiberContent'] -     # Fiber for fullness
        0.2 * df_scored['FatContent'] -       # Penalize high fat
        0.4 * df_scored['SugarContent'] -     # Strong penalty for sugar
        0.1 * df_scored['Calories']           # Moderate calorie penalty
    )
    
    # Filter out non-meal recipes (sauces, drinks, desserts)
    non_meals = ['Sauce', 'Beverage', 'Dessert', 'Drink', 'Condiment']
    if 'RecipeCategory' in df_scored.columns:
        df_scored = df_scored[~df_scored['RecipeCategory'].isin(non_meals)]
    
    # Normalize the final score to 0-1 range
    df_scored['WeightLossScore'] = (df_scored['WeightLossScore'] - df_scored['WeightLossScore'].min()) / \
                                  (df_scored['WeightLossScore'].max() - df_scored['WeightLossScore'].min())
    
    # Filter out recipes with extreme values (optional)
    df_scored = df_scored[
        (df_scored['Calories'] < 0.8) &  # Normalized threshold (~80th percentile)
        (df_scored['SugarContent'] < 0.7)
    ]
    
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

def recommend_meals_for_user(df, user_profile):
    """
    Recommend meals based on user's calorie goals and nutritional needs.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with recipe data and weight loss scores
    user_profile (dict): User profile dictionary with calorie goals and preferences
    
    Returns:
    pandas.DataFrame: Recommended meals with nutritional information
    """
    # Make a copy to avoid modifying the original
    df_recommended = df.copy()
    
    # Calculate daily meal calorie targets (assuming 3 meals + 2 snacks)
    meal_calorie_target = user_profile['CalorieGoal'] * 0.3  # 30% for main meals
    snack_calorie_target = user_profile['CalorieGoal'] * 0.1  # 10% for snacks
    
    # Filter recipes based on calorie content
    df_recommended = df_recommended[
        (df_recommended['Calories'] <= meal_calorie_target) &
        (df_recommended['WeightLossScore'] > 0.5)  # Only high-scoring recipes
    ]
    
    # Sort by weight loss score and calories
    df_recommended = df_recommended.sort_values(
        ['WeightLossScore', 'Calories'],
        ascending=[False, True]
    )
    
    # Select top recommendations
    recommendations = df_recommended.head(5)
    
    print("\n=== Meal Recommendations ===")
    print(f"Based on your daily calorie goal of {user_profile['CalorieGoal']} calories")
    print(f"Target calories per meal: {meal_calorie_target:.0f}")
    print("\nRecommended Meals:")
    
    for idx, meal in recommendations.iterrows():
        print(f"\n{meal['Name']}")
        print(f"Calories: {meal['Calories']:.0f}")
        print(f"Protein: {meal['ProteinContent']:.1f}g")
        print(f"Fiber: {meal['FiberContent']:.1f}g")
        print(f"Weight Loss Score: {meal['WeightLossScore']:.2f}")
    
    return recommendations

def main():
    # Read the CSV file
    df = pd.read_csv('recipes.csv')
    
    # Perform EDA and cleaning
    df_cleaned = perform_EDA(df)
    
    # Select features for feature engineering
    df_selected = select_features_for_feature_Engineering(df_cleaned)
    
    # Calculate weight loss score
    df_with_scores = calculate_weight_loss_score(df_selected)
    
    # Filter meal recipes
    df_filtered = filter_meal_recipes(df_with_scores)
    
    # Create user profile (example values)
    from user_profile import get_user_profile, print_user_profile
    
    user_profile = get_user_profile(
        age=20,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    
    # Print user profile
    print_user_profile(user_profile)
    
    # Get meal recommendations
    recommendations = recommend_meals_for_user(df_filtered, user_profile)

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

