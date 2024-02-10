import numpy as np
import random
from numpy import linalg
from sklearn.metrics.pairwise import pairwise_distances
import datetime

class Kmeans:
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=0.0001, n_init='auto', metric='cosine'):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric

    def __kmeans_random_centroids(self, X):
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)

        centroids = []
        for i in range(self.n_clusters):
            centroid = []
            for j in range(np.shape(X)[1]):
                centroid.append(random.uniform(min[j], max[j]))

            centroids.append(centroid)

        return centroids

    def __kmeans_plus_plus_centroids(self, X):
        centroids = []
        for i in np.random.randint(0,np.shape(X)[0], self.n_clusters):
            centroids.append(X[i])

        return centroids

    def __kmeans_calculate_distances(self, this_points, others_points):
        distances = []
        if (self.metric=='euclidean' or self.metric=='cosine'):
            distances = pairwise_distances(this_points, others_points, metric=self.metric)
        elif (self.metric=='mahalanobis'):
            '''points = np.vstack((this_points, others_points))

            distances = pairwise_distances(points, metric='mahalanobis', V = np.cov(points))

            distances = distances[0:len(this_points), len(this_points):]'''

            distances = pairwise_distances(this_points, others_points, metric='mahalanobis', V = np.cov(others_points))
        else:
            raise ValueError('Somente metrica euclidean, mahalanobis or cosine')

        return distances

    def __kmeans_distribution(self, X, centroids):
        distances = self.__kmeans_calculate_distances(centroids, X)
        min_distances = np.argmin(distances, axis=0)

        labels = {}
        inertia = 0
        for i in range(self.n_clusters):            
            labels[i] = list(np.where(min_distances == i)[0])
            if (len(labels[i])> 0):
                inertia += np.sum(np.power(distances[i][labels[i]], 2))

        return labels, inertia

    def __is_valid_first_centroid(self, labels):
        valid_centroid = True
        
        for i in labels:        
            if (len(labels[i])==0):
                valid_centroid = False
                break
        
        return valid_centroid
    
    def __kmeans_check(self, X, centroids, labels, inertia):
        if (not self.__is_valid_first_centroid(labels)):
            print('Centroid {0} Invalido, um novo sera recalculado'.format(len(centroids)))
            if self.init == 'random':        
                centroids = self.__kmeans_random_centroids(X)
            else:
                centroids = self.__kmeans_plus_plus_centroids(X)
            labels, inertia = self.__kmeans_distribution(X, centroids)
            centroids, labels, inertia = self.__kmeans_check(X, centroids, labels, inertia)

        return centroids, labels, inertia

    def __kmeans_with_initial_centroids(self, centroids, X):
        centroids_anterior = []
        for i in range(self.max_iter):
            labels, inertia = self.__kmeans_distribution(X, centroids)

            #If it is the first iteration, check if the centroid is valid,
            #if it is not recalculated
            if (i == 0):
                centroids, labels, inertia = self.__kmeans_check(X, centroids, labels, inertia)

            result = [centroids.copy(), labels.copy(), inertia]
            centroids_anterior = centroids.copy()
            for j in labels:
                if np.shape(labels[j])[0] > 0: 
                    avg = np.average(np.take(X, labels[j], axis=0), axis = 0)
                    centroids[j] = avg

            try:
                distances = self.__kmeans_calculate_distances(centroids_anterior, centroids)
                
                limit_stop = (distances[np.where(np.identity(self.n_clusters) == 1)] <= self.tol)
            except linalg.LinAlgError:
                limit_stop = (np.ones(self.n_clusters, dtype=int) == 1)
                print('linalg.LinAlgError')
            print(i, np.sum(limit_stop), datetime.datetime.now(), flush=True)
            if (np.sum(limit_stop) == self.n_clusters):
                break

        return result


    def fit(self, X):
        if self.init == 'random':   
            centroids = self.__kmeans_random_centroids(X)
        else:
            centroids = self.__kmeans_plus_plus_centroids(X)

        result = self.__kmeans_with_initial_centroids(centroids, X)

        centroids = result[0]
        labels = result[1]
        
        self.cluster_centers_ = centroids
        self.inertia_ = result[2]

        self.labels_ = np.full(np.shape(X)[0], -1, dtype=np.int32)
        for i in labels:
            np.put(self.labels_, labels[i], i)

        return self

def predict(self, X):    
    distances = self.__kmeans_calculate_distances(self.cluster_centers_, X)
    labels = np.argmin(distances, axis=0)

    return labels
