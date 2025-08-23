import pandas as pd
import numpy as np

def generate_meal_data():
    """Generate sample meal data from recipes.csv for GraphQL API"""
    try:
        # Read the recipes.csv file
        print("Reading recipes.csv...")
        df = pd.read_csv('recipes.csv')
        print(f"Total meals found: {len(df)}")
        
        # Select relevant columns for the API
        columns_to_keep = [
            'RecipeId', 'Name', 'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 
            'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeServings'
        ]
        
        # Filter columns that exist in the dataset
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df_selected = df[available_columns].copy()
        
        # Clean the data
        df_selected = df_selected.dropna(subset=['Calories', 'ProteinContent'])
        
        # Convert numeric columns
        numeric_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                          'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
        
        for col in numeric_columns:
            if col in df_selected.columns:
                df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        
        # Remove rows with extreme values (outliers)
        df_cleaned = df_selected[
            (df_selected['Calories'] > 0) & 
            (df_selected['Calories'] < 2000) &
            (df_selected['ProteinContent'] > 0) &
            (df_selected['ProteinContent'] < 100)
        ]
        
        # Calculate a simple health score
        df_cleaned['HealthScore'] = (
            (df_cleaned['ProteinContent'] * 4) +  # Protein is good
            (df_cleaned['FiberContent'] * 2) +    # Fiber is good
            (df_cleaned['Calories'] * -0.1) +     # Lower calories is better
            (df_cleaned['FatContent'] * -0.5) +   # Lower fat is better
            (df_cleaned['SugarContent'] * -0.3)   # Lower sugar is better
        )
        
        # Classify meals as good (1) or not good (0) based on health score
        median_score = df_cleaned['HealthScore'].median()
        df_cleaned['IsGoodMeal'] = (df_cleaned['HealthScore'] > median_score).astype(int)
        
        # Select top 100 meals for the API
        df_final = df_cleaned.nlargest(100, 'HealthScore')
        
        # Save to CSV
        output_file = 'df_combinedUser_data_sample.csv'
        df_final.to_csv(output_file, index=False)
        
        print(f"✅ Generated {len(df_final)} meals and saved to {output_file}")
        print(f"Good meals: {(df_final['IsGoodMeal'] == 1).sum()}")
        print(f"Regular meals: {(df_final['IsGoodMeal'] == 0).sum()}")
        
        # Show sample data
        print("\nSample meals:")
        print(df_final[['Name', 'Calories', 'ProteinContent', 'FatContent', 'HealthScore', 'IsGoodMeal']].head())
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating meal data: {e}")
        return None

if __name__ == "__main__":
    generate_meal_data() 