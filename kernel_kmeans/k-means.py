import numpy as np
import matplotlib.pyplot as plt
from k1_freq_vec import k1
from k2_n_gram import k2

# impelment k-means clustering from scratch, but instead of eucledian distance netween points, we shall use alpha * k1 + (1 - alpha) * k2, where alpha is a hyperparameter

# Define the Kernelized KMeans class with cosine similarity
class KMeans:
    # Initialize the class
    def __init__(self, k=3, max_iter=100, alpha=0.5):
        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self.centroids = None
    
    def initialize_centroids(self, docs): # docs contains list of documents (their path)
        # Randomly choose k data points as centroids
        lst = [i for i in range(len(docs))]
        np.random.shuffle(lst)
        self.centroids = lst[:self.k]
        print(self.centroids)

    def compute_distance(self, doc1, doc2):
        # Compute the distance between two documents
        return self.alpha * k1(doc1, doc2) + (1 - self.alpha) * k2(doc1, doc2)
    
    def find_closest_centroids(self, docs):
        # Compute the distance between each document and the centroids and assign the document to the closest centroid
        closest_centroids = []
        for doc in docs:
            distances = []
            for centroid in self.centroids:
                distances.append(self.compute_distance(doc, docs[centroid]))
            closest_centroids.append(self.centroids[np.argmax(distances)]) # np.argmax returns the index of the maximum value in the array
        return closest_centroids
    
    def fit(self, docs):
        # Run the KMeans algorithm
        self.initialize_centroids(docs)
        for i in range(self.max_iter):
            closest_centroids = self.find_closest_centroids(docs)

            new_centroids = []
            for j in range(self.k):
                # for each cluster, calculate the center of gravity of the documents in the cluster
                # take the distance of each point from all the other points in the cluster and the add the distance, one with the minimum distance is the center of gravity
                curr_centroid = self.centroids[j]
                new_centroid = None
                max_similarity = 0
                cluster_points = []
                for point in range(len(closest_centroids)):
                    if closest_centroids[point] == curr_centroid:
                        cluster_points.append(point)
    
                for point1 in cluster_points:
                    sum_similarity = 0
                    for point2 in cluster_points:
                        if point1 != point2:
                            sum_similarity += self.compute_distance(docs[point1], docs[point2])
                    curr_similarity = sum_similarity / ((len(cluster_points) - 1) if len(cluster_points) > 1 else 1)
                    if curr_similarity >= max_similarity:
                        max_similarity = curr_similarity
                        new_centroid = point1

                new_centroids.append(new_centroid)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids
    
    def display_clusters(self, docs):
        # Plot the clusters and the centroids
        closest_centroids = self.find_closest_centroids(docs)
        centroid_doc_map = {}
        for i in range(len(closest_centroids)):
            if closest_centroids[i] not in centroid_doc_map.keys():
                centroid_doc_map[closest_centroids[i]] = []
            centroid_doc_map[closest_centroids[i]].append(docs[i])
        for centroid in centroid_doc_map.keys():
            print('Cluster {}: {}'.format(centroid, centroid_doc_map[centroid]))

        from error import get_entropy
        print("Entropy :", get_entropy(centroid_doc_map))
        print("Accuracy :", (1 - get_entropy(centroid_doc_map)) * 100, "%" )


alpha = 0.3

# Test the KMeans class
# Initialize the data
docs = ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt', 'D5.txt', 'D6.txt', 'D7.txt', 'D8.txt', 'D9.txt', 'D10.txt']
# Create an instance of KMeans
kmeans = KMeans(k=4, alpha=alpha)
# Train the KMeans clustering model
kmeans.fit(docs)
# Plot the clusters
kmeans.display_clusters(docs)
    