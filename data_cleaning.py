import pandas as pd
import numpy as np

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

def check_data_quality(df):
    """
    Perform additional data quality checks on the cleaned DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The cleaned DataFrame
    
    Returns:
    dict: Dictionary containing data quality metrics
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    print("\nData Quality Metrics:")
    print(f"Total Rows: {quality_metrics['total_rows']}")
    print(f"Total Columns: {quality_metrics['total_columns']}")
    print(f"Duplicate Rows: {quality_metrics['duplicate_rows']}")
    print("\nNumeric Columns:", quality_metrics['numeric_columns'])
    print("\nCategorical Columns:", quality_metrics['categorical_columns'])
    
    return quality_metrics

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('recipes.csv')
    df_cleaned = perform_EDA(df)
    quality_metrics = check_data_quality(df_cleaned) 