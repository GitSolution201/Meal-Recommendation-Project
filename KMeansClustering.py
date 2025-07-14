import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the meal data
df = pd.read_csv('classified_meals.csv')

# Select features for clustering: Calories, ProteinContent, FiberContent
features = ['Calories', 'ProteinContent', 'FiberContent']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose number of clusters (e.g., 3 for high/medium/low)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['NutriCluster'] = kmeans.fit_predict(X_scaled)

# Print cluster statistics
print('Number of meals in each cluster:')
print(df['NutriCluster'].value_counts())
print('\nMean nutritional values for each cluster:')
print(df.groupby('NutriCluster')[features].mean())

# Visualize clusters in 2D (Calories vs ProteinContent, colored by cluster)
plt.figure(figsize=(8,6))
plt.scatter(df['Calories'], df['ProteinContent'], c=df['NutriCluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Calories')
plt.ylabel('ProteinContent')
plt.title('K-Means Clusters: Calories vs ProteinContent')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show() 