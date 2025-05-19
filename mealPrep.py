import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def select_features(df):
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

def main():
    # Read the CSV file
    df = pd.read_csv('recipes.csv')
    
    # Print describe statistics before EDA
    print("\nStatistics BEFORE EDA:")
    print(df.describe())
    
    # Perform EDA and cleaning
    df_cleaned = perform_EDA(df)
    
    # Select features
    df_selected = select_features(df_cleaned)
    
    # Print describe statistics after feature selection
    print("\nStatistics AFTER Feature Selection:")
    print(df_selected.describe())
    
    # Create correlation heatmap
    create_correlation_heatmap(df_selected)

if __name__ == "__main__":
    main()


# Display the updated column names after dropping RecipeYield
print("\nUpdated Column Names after dropping RecipeYield:")

