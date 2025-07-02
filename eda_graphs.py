import matplotlib.pyplot as plt
import seaborn as sns

def EDA_graphs(df_cleaned):
    # Pie chart: total calories by top 10 categories
    top10_categories = df_cleaned['RecipeCategory'].value_counts().head(10).index
    calories_by_category = df_cleaned[df_cleaned['RecipeCategory'].isin(top10_categories)].groupby('RecipeCategory')['Calories'].sum()
    plt.figure(figsize=(7, 7))
    plt.pie(calories_by_category, labels=calories_by_category.index, autopct='%1.1f%%', startangle=140)
    plt.title('Total Calories by Top 10 Recipe Categories')
    plt.axis('equal')
    plt.show()

    # Bar chart for top 20 RecipeCategory: x-axis = categories, y-axis = count
    category_counts = df_cleaned['RecipeCategory'].value_counts().head(20)
    plt.bar(category_counts.index, category_counts.values)
    plt.title('Top 20 Recipe Category Counts')
    plt.xlabel('Recipe Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Heatmap for nutritional features
    nutrition_cols = [
        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
    ]
    corr = df_cleaned[nutrition_cols].corr()
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Nutritional Features')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() 