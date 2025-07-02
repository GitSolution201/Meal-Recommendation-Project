import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_recommend_meals(df_filtered, user_profile, knn_features=None, n_neighbors=5):
    if knn_features is None:
        knn_features = ['Calories', 'ProteinContent', 'FatContent', 'FiberContent', 'WeightLossScore']
    X = df_filtered[knn_features]
    user_vector = np.array([
        user_profile['CalorieGoal'] * 0.3,  # per meal
        75,  # target protein per meal (example)
        10,  # target fat per meal (example)
        100,   # target fiber per meal (example)
        1.0  # ideal weight loss score
    ]).reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan')
    knn.fit(X)
    distances, indices = knn.kneighbors(user_vector)
    knn_recommendations = df_filtered.iloc[indices[0]]
    return knn_recommendations 