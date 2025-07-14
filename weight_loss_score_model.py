# randomForestModel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load classified meals data
meals = pd.read_csv('classified_meals.csv')

# Add user features (example values, replace with actual user data as needed)
meals['BMI'] = 25  # Replace with actual BMI values if available
meals['BMR'] = 1500  # Replace with actual BMR values if available
meals['age'] = 35  # Replace with actual age values if available
meals['weight_kg'] = 70  # Replace with actual weight values if available

# Features to use (all available nutrition columns + user features)
features = [
    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
    'BMI', 'BMR', 'age', 'weight_kg'
]

X = meals[features]
y = meals['IsGoodMeal']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Example: Predict on new meal (replace with actual values)
# new_meal = pd.DataFrame([{ ... }])
# print('Predicted IsGoodMeal:', clf.predict(new_meal)[0]) 