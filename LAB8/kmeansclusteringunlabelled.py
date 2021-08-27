# 1. Generate data and convert it to a pandas DataFrame
# Imports
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, n_features=2,
cluster_std=0.2, random_state=0)

#2. Basic Data Visualization
# Scatter plot of the data points
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, -1])
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.show()

#3. Using scikit-learn to perform K-Means clustering
# Using scikit-learn to perform K-Means clustering
from sklearn.cluster import KMeans
# Specify the number of clusters (3) and fit the data X
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

#4. Visualize and evaluate the results
# Get the cluster centroids
kmeans.cluster_centers_

# Get the cluster labels
kmeans.labels_

# Plotting the cluster centers and the data points on a 2D plane
plt.scatter(X[:, 0], X[:, -1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],c='red', marker='x')
plt.title('Data points and cluster centroids')
plt.show()