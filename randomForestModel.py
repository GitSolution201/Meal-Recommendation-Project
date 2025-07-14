# randomForestModel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

# Load classified meals data
meals = pd.read_csv('classified_meals.csv')

# Add user features (example values, replace with actual user data as needed)
# These lines should be removed if you do not have actual user data for these features.
# Setting them to constant values for all samples does not add any useful information to your model and can even harm its performance or create misleading results.
# If you have real user data for these features, use the actual values.
# If you do not have this data, simply remove these lines from your code.

# Features to use (all available nutrition columns + user features)
features = [
    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
    # 'BMI', 'BMR', 'age', 'weight_kg'
]

X = meals[features]
y = meals['IsGoodMeal']

# Train/test split
# ensures both the sets have same calls proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Train classification model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save predictions for the 20% test data to CSV
results = X_test.copy()
results['Name'] = meals.loc[X_test.index, 'Name']  # Add meal name
results['TrueLabel'] = y_test.values
results['PredictedLabel'] = y_pred
results.to_csv('random_forest_test_predictions.csv', index=False)
print('Test predictions saved to random_forest_test_predictions.csv')

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calculate mean and std for train and test scores
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.title('Learning Curve for Random Forest')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

# After fitting your model (e.g., clf)
importances = clf.feature_importances_
feature_names = X.columns  # or your list of features

# Create a DataFrame for easy plotting
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances in Random Forest')
plt.gca().invert_yaxis()
plt.show()

# After you have predictions (e.g., y_pred = clf.predict(X_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show() 