### Given a normalised two-dimensional set of vectors, perform classical k means clustering into number_of_clusters clusters
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def knn_clustering(data, number_of_clusters):
    knn = NearestNeighbors(n_neighbors=number_of_clusters)
    knn.fit(data)
    distances, indices = knn.kneighbors(data)
    cluster_labels = indices[:, 0]
    cluster_centers = np.array([data[idx].mean(axis=0) for idx in indices])
    return cluster_labels, cluster_centers

def kmeans_clustering(data, number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

# Example usage
data = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9], [0.1, 0.2], [0.3, 0.4],
                 [0.7, 0.6], [0.9, 0.8], [1.2, 1.3], [1.4, 1.5], [1.6, 1.7], [1.8, 1.9],
                 [2.0, 2.1], [2.2, 2.3], [2.4, 2.5], [2.6, 2.7], [2.8, 2.9], [3.0, 3.1],
                 [3.2, 3.3], [3.4, 3.5], [3.6, 3.7], [3.8, 3.9], [4.0, 4.1], [4.2, 4.3],
                 [4.4, 4.5], [4.6, 4.7], [4.8, 4.9], [5.0, 5.1], [5.2, 5.3], [5.4, 5.5]])

normalized_data = normalize(data)  # Normalize by dividing each vector by its length
number_of_clusters = 3

cluster_labels, cluster_centers = knn_clustering(normalized_data, number_of_clusters)

# Plotting the clusters
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-NN Clustering')
plt.show()

data = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8], [0.8, 0.9, 1.0], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.7, 0.6, 0.5], [0.9, 0.8, 0.7], [1.2, 1.3, 1.4], [1.4, 1.5, 1.6], [1.6, 1.7, 1.8], [1.8, 1.9, 2.0]])

normalized_data = normalize(data)  # Normalize by dividing each vector by its length
number_of_clusters = 3

cluster_labels, cluster_centers = knn_clustering(normalized_data, number_of_clusters)


# Plotting the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2], c=cluster_labels, cmap='viridis')
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('k-NN Clustering')
plt.show()










