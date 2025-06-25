def get_user_profile(age, gender, weight_kg, height_cm, activity_level="moderate", goal="moderate"):
    """
    Calculates BMR, BMI, and recommended daily calorie intake for weight loss.
    BMR is the number of calories your body needs to maintain basic physiological functions at rest
    BMI is a numerical value calculated from your height and weight, used to assess whether you are underweight, normal weight, overweight, or obese
    Parameters:
    - age (int): Age in years
    - gender (str): 'male' or 'female'
    - weight_kg (float): Weight in kilograms
    - height_cm (float): Height in centimeters
    - activity_level (str): 'sedentary', 'light', 'moderate', 'active', or 'very_active'
    - goal (str): 'mild', 'moderate', 'aggressive'
    TDEE is the total number of calories your body burns in a day, including all activities (not just at rest).
    Returns:
    - dict: User profile including BMI, BMR, TDEE, and calorie goal
    """
    # 1. Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # 2. Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # 3. Adjust for Activity Level
    activity_factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    tdee = bmr * activity_factors.get(activity_level.lower(), 1.55)

    # 4. Adjust for Weight Loss Goal
    deficit_factors = {
        'mild': 0.9,       # ~10% deficit
        'moderate': 0.8,   # ~20% deficit
        'aggressive': 0.7  # ~30% deficit
    }
    calorie_goal = tdee * deficit_factors.get(goal.lower(), 0.8)

    return {
        'BMI': round(bmi, 2),
        'BMR': round(bmr),
        'TDEE': round(tdee),
        'CalorieGoal': round(calorie_goal),
        'ActivityLevel': activity_level,
        'WeightGoal': goal
    }

def get_bmi_category(bmi):
    """
    Returns the BMI category based on the BMI value.
    
    Parameters:
    - bmi (float): Body Mass Index value
    
    Returns:
    - str: BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_activity_level_description(activity_level):
    """
    Returns a description of the activity level.
    
    Parameters:
    - activity_level (str): Activity level key
    
    Returns:
    - str: Description of the activity level
    """
    descriptions = {
        'sedentary': "Little or no exercise, desk job",
        'light': "Light exercise 1-3 days/week",
        'moderate': "Moderate exercise 3-5 days/week",
        'active': "Hard exercise 6-7 days/week",
        'very_active': "Very hard exercise, physical job or training twice per day"
    }
    return descriptions.get(activity_level.lower(), "Moderate exercise 3-5 days/week")

def get_weight_goal_description(goal):
    """
    Returns a description of the weight loss goal.
    
    Parameters:
    - goal (str): Weight loss goal key
    
    Returns:
    - str: Description of the weight loss goal
    """
    descriptions = {
        'mild': "Mild weight loss (0.25-0.5 kg per week)",
        'moderate': "Moderate weight loss (0.5-0.75 kg per week)",
        'aggressive': "Aggressive weight loss (0.75-1 kg per week)"
    }
    return descriptions.get(goal.lower(), "Moderate weight loss (0.5-0.75 kg per week)")

def print_user_profile(profile):
    """
    Prints a formatted user profile with all relevant information.
    
    Parameters:
    - profile (dict): User profile dictionary
    """
    print("\n=== User Profile ===")
    print(f"BMI: {profile['BMI']} ({get_bmi_category(profile['BMI'])})")
    print(f"BMR (Basal Metabolic Rate): {profile['BMR']} calories/day")
    print(f"TDEE (Total Daily Energy Expenditure): {profile['TDEE']} calories/day")
    print(f"Daily Calorie Goal: {profile['CalorieGoal']} calories/day")
    print(f"\nActivity Level: {profile['ActivityLevel'].title()}")
    print(f"Description: {get_activity_level_description(profile['ActivityLevel'])}")
    print(f"\nWeight Loss Goal: {profile['WeightGoal'].title()}")
    print(f"Description: {get_weight_goal_description(profile['WeightGoal'])}")

if __name__ == "__main__":
    # Example usage
    profile = get_user_profile(
        age=30,
        gender="male",
        weight_kg=80,
        height_cm=175,
        activity_level="moderate",
        goal="moderate"
    )
    print_user_profile(profile) 