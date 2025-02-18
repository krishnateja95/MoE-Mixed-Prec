import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Your list of values
values = np.array([1, 2, 3, 4, 5, 6, 7, 0.8, 9]).reshape(-1, 1)

# Apply k-means clustering with k=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(values)

# Get the cluster labels for each value
labels = kmeans.labels_

# Print the clusters
for i, label in enumerate(labels):
    print(f"Value {values[i][0]} is in cluster {label}")

# Plot the clusters
plt.scatter(values, np.zeros(len(values)), c=labels)
plt.scatter(kmeans.cluster_centers_, np.zeros(len(kmeans.cluster_centers_)), marker="X", s=200, c='red')
plt.show()
