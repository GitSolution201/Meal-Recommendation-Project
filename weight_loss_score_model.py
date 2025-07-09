import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load meal data
meals = pd.read_csv('meals.csv')

# Apply the WeightLossScore formula
meals['WeightLossScore'] = (
    0.5 * meals['ProteinContent'] +
    0.3 * meals['FiberContent'] -
    0.2 * meals['FatContent'] -
    0.1 * meals['SugarContent'] -
    0.05 * meals['Calories']
)

# Features to use (all available nutrition columns)
features = [
    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
]

# Prepare data
X = meals[features]
y = meals['WeightLossScore']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# Example: Predict on new meal
# new_meal = pd.DataFrame([{ ... }])
# print('Predicted WeightLossScore:', reg.predict(new_meal)[0]) 