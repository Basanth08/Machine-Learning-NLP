import pandas as pd
import numpy as np

# This is a Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# This is the Function to determine the unique clusters in the data
def determine_clusters(data):
    return set(data['Cluster'])

# This is a Function to calculate cohesion (ai) for a given cluster
def calculate_cohesion(data, cluster):
    cluster_data = data[data['Cluster'] == cluster]
    cohesion = []
    for _, instance in cluster_data.iterrows():
        distances = [euclidean_distance(instance[1:-1], other[1:-1]) for _, other in cluster_data.iterrows() if not instance.equals(other)]
        if len(distances) > 0:
            cohesion.append(np.mean(distances))
        else:
            cohesion.append(0)
    return cohesion

# This is a Function to calculate separation (bi) for a given cluster
def calculate_separation(data, clusters, cluster):
    separation = []
    for _, instance in data[data['Cluster'] == cluster].iterrows():
        sep = float('inf')
        for other_cluster in clusters:
            if other_cluster != cluster:
                other_cluster_data = data[data['Cluster'] == other_cluster]
                distances = [euclidean_distance(instance[1:-1], other[1:-1]) for _, other in other_cluster_data.iterrows()]
                if len(distances) > 0:
                    sep = min(sep, np.mean(distances))
        separation.append(sep)
    return separation

# This is a Function to calculate the silhouette coefficient (si) for each data point
def calculate_silhouette(data):
    clusters = determine_clusters(data)
    silhouette_coefficients = []
    for cluster in clusters:
        cohesion = calculate_cohesion(data, cluster)
        separation = calculate_separation(data, clusters, cluster)
        for ai, bi in zip(cohesion, separation):
            if max(ai, bi) > 0:
                si = abs(1 - (ai / bi))
            else:
                si = 0
            silhouette_coefficients.append(si)
    return silhouette_coefficients

# This Reads the data files and calculates the average silhouette coefficient for each k
k_values = [2, 3, 5, 7]
for k in k_values:
    file_path = '/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/k' + str(k) + '.csv'
    data = pd.read_csv(file_path)
    silhouette_coefficients = calculate_silhouette(data)
    avg_silhouette = sum(silhouette_coefficients) / len(silhouette_coefficients)
    print("Average silhouette coefficient for k = " + str(k) + " is: " + str(round(avg_silhouette, 3)))