import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
data = np.load('Data_for_Cluster.npz')
X = data['X']
labels_true = data['labels_true']

# K-Means model
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Silhouette Scores
silhouette_score_kmeans = silhouette_score(X, kmeans_labels)
silhouette_score_dbscan = silhouette_score(X, dbscan_labels)

# Plotting
plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clustering\nSilhouette Score: {:.2f}".format(silhouette_score_kmeans))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# DBSCAN plot
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
plt.title("DBSCAN Clustering\nSilhouette Score: {:.2f}".format(silhouette_score_dbscan))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
