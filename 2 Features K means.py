import numpy as np
import matplotlib.pyplot as plt

# Provided 2D dataset
data = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6],
    [9, 11], [8, 2], [10, 2], [9, 3], [4, 9]
])

k = 2
max_iters = 100

# Step 1: Random initialization of centroids
np.random.seed(42)
centroids = data[np.random.choice(len(data), k, replace=False)]

for iteration in range(max_iters):
    # Step 2: Assign clusters based on closest centroid
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)

    # Step 3: Calculate new centroids
    new_centroids = np.array([data[cluster_assignments == j].mean(axis=0) for j in range(k)])
    
    # Check convergence
    if np.allclose(centroids, new_centroids):
        print(f"Converged at iteration {iteration + 1}")
        break
    centroids = new_centroids

# Final assignments and centroids
print("\nFinal Cluster Assignments:")
for idx, point in enumerate(data):
    print(f"Point {point} -> Cluster {cluster_assignments[idx]}")

print("\nFinal Centroids:")
print(centroids)

# WCSS (Within-cluster sum of squares)
wcss = 0
for j in range(k):
    cluster_points = data[cluster_assignments == j]
    wcss += np.sum((cluster_points - centroids[j]) ** 2)

print(f"\nWCSS (Within-Cluster Sum of Squares): {wcss:.4f}")

# Visualization
colors = ['red', 'blue']
for j in range(k):
    cluster_points = data[cluster_assignments == j]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[j], label=f'Cluster {j}')
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroids')
plt.title("k-Means Clustering (k=2)")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.grid(True)
plt.show()
