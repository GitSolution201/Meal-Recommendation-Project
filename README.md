# Meal Recommendation System

An intelligent machine learning-based meal recommendation system that classifies meals as "good" or "bad" based on nutritional analysis and provides personalized dietary suggestions for weight management.

## 🚀 Quick Start

### First Time Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Project**
   ```bash
   python mealPrep.py
   ```
   This will automatically run:
   - Exploratory Data Analysis (EDA)
   - Feature Engineering
   - K-Nearest Neighbors (KNN) implementation

### Individual Model Execution

After running the main project, you can run individual machine learning models:

#### Support Vector Machine (SVM) Classifier
```bash
python SVMClassifier.py
```

#### Random Forest Classifier
```bash
python randomForestModel.py
```

## 📁 Project Structure

```
Project/
├── mealPrep.py                    # Main entry point - runs EDA, feature engineering, and KNN
├── SVMClassifier.py               # Support Vector Machine implementation
├── randomForestModel.py           # Random Forest Classifier implementation
├── KMeansClustering.py            # K-Means clustering for meal categorization
├── feature_engineering.py         # Feature engineering and synthetic user profiles
├── data_cleaning.py               # Data preprocessing and cleaning
├── eda_graphs.py                  # Exploratory data analysis and visualizations
├── knn_recommender.py             # K-Nearest Neighbors recommendation system
├── weight_loss_score_model.py     # Custom nutritional scoring system
├── user_profile.py                # User profile generation and management
├── feedback.py                    # User feedback system
├── generate_meal_data.py          # Meal data generation utilities
├── meal_recommendations.py        # Core recommendation engine
├── requirements.txt               # Python dependencies
├── package.json                   # Node.js dependencies (if applicable)
└── Report.pdf                     # Project documentation
```

## 🧠 Machine Learning Models

### 1. Random Forest Classifier
- **Purpose**: Primary meal classification model
- **Performance**: ~84.8% cross-validation accuracy
- **Features**: 10 nutritional components + custom WeightLossScore
- **Output**: Binary classification (good/bad meals)

### 2. Support Vector Machine (SVM)
- **Purpose**: Alternative classification approach
- **Performance**: ~70% accuracy
- **Features**: Same nutritional feature set
- **Output**: Binary classification with different decision boundaries

### 3. K-Nearest Neighbors (KNN)
- **Purpose**: Similarity-based meal recommendations
- **Features**: Nutritional similarity matching
- **Output**: Personalized meal suggestions based on user preferences

## 🔧 Features

- **Intelligent Meal Classification**: Automatically categorizes meals based on nutritional analysis
- **Custom WeightLossScore**: Proprietary algorithm for health optimization scoring
- **Synthetic User Profiles**: Generates diverse user scenarios for robust model training
- **Multi-Algorithm Approach**: Combines different ML techniques for comprehensive analysis
- **Feature Importance Analysis**: Identifies key nutritional drivers in meal quality
- **Learning Curve Analysis**: Monitors model performance and detects overfitting
- **Cross-Validation**: Ensures robust and reliable model evaluation

## 📊 Data Requirements

The system processes nutritional data including:
- Calories, Fat Content, Saturated Fats
- Cholesterol, Sodium, Carbohydrates
- Fiber, Sugar, Protein Content
- Custom WeightLossScore

## 🎯 Use Cases

- **Weight Management**: Personalized meal recommendations for health goals
- **Nutritional Education**: Understanding meal quality factors
- **Dietary Planning**: Intelligent meal selection based on preferences
- **Health Monitoring**: Tracking nutritional intake and meal quality

## 📈 Performance Metrics

- **Accuracy**: Overall classification performance
- **Precision/Recall**: Balanced performance across meal categories
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: Robust performance assessment
- **Feature Importance**: Nutritional factor ranking

## 🚨 Important Notes

- **Large Files**: CSV data files are not included in git (use Git LFS if needed)
- **Dependencies**: Ensure all requirements are installed before running
- **Python Version**: Compatible with Python 3.8+
- **Memory**: Minimum 8GB RAM recommended for large datasets

## 🔍 Troubleshooting

### Common Issues:
1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Memory Issues**: Reduce dataset size or increase system memory
3. **File Not Found**: Ensure all Python files are in the same directory

### Getting Help:
- Check the `Report.pdf` for detailed project documentation
- Review individual model files for specific implementation details
- Ensure all dependencies are properly installed

## 📝 License

This project is developed for academic purposes as part of a university placement report.

## 👨‍💻 Author

Tayyab Jamil - University Placement Project

---

**Happy Meal Planning! 🍽️**
