import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_recommend_meals(df_filtered, user_profile, knn_features=None, n_neighbors=3):
    if knn_features is None:
        knn_features = [
    'Calories',
    'FatContent',
    'SaturatedFatContent',
    'CholesterolContent',
    'SodiumContent',
    'CarbohydrateContent',
    'FiberContent',
    'SugarContent',
    'ProteinContent'
]
    X = df_filtered[knn_features]

    #X represents all the meals in the dataset 
    user_vector = np.array([
    user_profile['CalorieGoal'] * 0.3,  # Calories
    10,   # FatContent (example)
    3,    # SaturatedFatContent (example)
    50,   # CholesterolContent (example)
    500,  # SodiumContent (example)
    60,   # CarbohydrateContent (example)
    8,    # FiberContent (example)
    10,   # SugarContent (example)
    25    # ProteinContent (example)
]).reshape(1, -1)

    #creating a perfect target meal profile for the user 

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan')
    #n_neighbors is the number of neighbors to consider for the recommendation
    # here we are calculating the distancd between the user_vector and the meals in the dataset (X)
    knn.fit(X)

    distances, indices = knn.kneighbors(user_vector)

    # here we are returning the meals that are closest to the user_vector
    knn_recommendations = df_filtered.iloc[indices[0]]
    
    return knn_recommendations 