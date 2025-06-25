import pandas as pd

def show_best_worst_meals(df):
    """
    Display the best and worst 20 meals based on weight loss score.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with recipe data and weight loss scores
    """
    # Sort by weight loss score
    df_sorted = df.sort_values('WeightLossScore', ascending=False)
    
    # Get best and worst 20 meals
    best_meals = df_sorted.head(20)
    worst_meals = df_sorted.tail(20)
    
    print("\n=== Best 20 Meals for Weight Loss ===")
    for idx, meal in best_meals.iterrows():
        print(f"\n{meal['Name']}")
        print(f"Calories: {meal['Calories']:.0f}")
        print(f"Protein: {meal['ProteinContent']:.1f}g")
        print(f"Fiber: {meal['FiberContent']:.1f}g")
        print(f"Weight Loss Score: {meal['WeightLossScore']:.2f}")
    
    print("\n=== Worst 20 Meals for Weight Loss ===")
    for idx, meal in worst_meals.iterrows():
        print(f"\n{meal['Name']}")
        print(f"Calories: {meal['Calories']:.0f}")
        print(f"Protein: {meal['ProteinContent']:.1f}g")
        print(f"Fiber: {meal['FiberContent']:.1f}g")
        print(f"Weight Loss Score: {meal['WeightLossScore']:.2f}")

#For example, if the user's daily goal is 2000 calories, and 30% is allocated for a main meal, the target per meal is 600 calories.
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
    
    print(f"\nInitial dataset size: {len(df_recommended)} recipes")
    #The function first tries to recommend meals that strictly fit the user's calorie target for a meal. However, sometimes there may be no recipes that meet both the calorie and weight loss score criteria
    # Filter recipes based on calorie content
    df_recommended = df_recommended[
        (df_recommended['Calories'] <= meal_calorie_target) &
        (df_recommended['WeightLossScore'] > 0.5)  # Only high-scoring recipes
    ]
    
    print(f"After initial filtering: {len(df_recommended)} recipes")
    
    # If no meals meet the criteria, relax the constraints
    if len(df_recommended) == 0:
        print("\nNo meals found with strict criteria. Relaxing constraints...")
        df_recommended = df.copy()
        
        # First try with just calorie constraint
        df_recommended = df_recommended[df_recommended['Calories'] <= meal_calorie_target * 1.5]
        print(f"After calorie filtering: {len(df_recommended)} recipes")
        
        # Then sort by weight loss score
        df_recommended = df_recommended.sort_values('WeightLossScore', ascending=False)
    
    # Sort by weight loss score and calories
    df_recommended = df_recommended.sort_values(
        ['WeightLossScore', 'Calories'],
        ascending=[False, True]
    )
    
    # Select top 5 recommendations
    recommendations = df_recommended.head(5)
    
    print("\n=== Top 5 Recommended Meals ===")
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

def recommend_meals_knn(df, user_profile, n_recommendations=5):
    """
    Recommend meals using K-Nearest Neighbors based on user's nutritional targets.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with meal data and nutritional features
    user_profile (dict): User profile dictionary with calorie goal and preferences
    n_recommendations (int): Number of meals to recommend
    
    Returns:
    pandas.DataFrame: Top recommended meals
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    knn_features = ['Calories', 'ProteinContent', 'FatContent', 'FiberContent', 'WeightLossScore']
    X = df[knn_features]
    
    # Create user vector based on profile and example targets
    user_vector = np.array([
        user_profile['CalorieGoal'] * 0.3,  # per meal
        25,  # target protein per meal (example)
        10,  # target fat per meal (example)
        8,   # target fiber per meal (example)
        1.0  # ideal weight loss score
    ]).reshape(1, -1)
    
    knn = NearestNeighbors(n_neighbors=n_recommendations, metric='euclidean')
    knn.fit(X)
    distances, indices = knn.kneighbors(user_vector)
    recommendations = df.iloc[indices[0]]
    return recommendations

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from user_profile import get_user_profile
    
    # Create sample data
    df = pd.read_csv('recipes.csv')
    
    # Create example user profile
    profile = get_user_profile(
        age=30,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    
    # Get personalized meal recommendations
    recommend_meals_for_user(df, profile) 