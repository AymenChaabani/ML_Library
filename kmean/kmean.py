# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:39:40 2022

@author: aymen
"""

import numpy as np

class kmean():
    def converge(self,A,B,threshold):
        #this function return whether the 
        dist_matrix=np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                dist_matrix[i,j]=self.euclidian_distance(A[i],B[j])
        return  np.max(dist_matrix.diagonal()) <= threshold

        
    def get_centroids_random(self,features,k):
        #this function return random differents centroids given an input k
        num_features=features.shape[0]
        centroid_index=np.random.choice(num_features,k,replace=False)
        #ventroids points
        centroid=[]
        for i in centroid_index:
            centroid.append(features[i])
        centroid_unique=list(centroid)
        while len(centroid_unique)< k :
            centroid_index=np.random.choice(num_features,k,replace=False)
            centroid=[]
            for i in centroid_index:
                centroid.append(features[i])
            centroid_unique=list(centroid)
        centroids=np.array(centroid)
        return centroids
    def euclidian_distance(self,A,B):
        #this function calculates the euclidian distance between two elements
        return np.linalg.norm(B - A)
    def clusters(self,features,centroids):
        #this function return clasters given an input feature data and centroids
        k=centroids.shape[0]
        dist_matrix=np.zeros((features.shape[0], centroids.shape[0]))
        for i in range(features.shape[0]):
            for j in range(centroids.shape[0]):
                dist_matrix[i,j]=self.euclidian_distance(features[i],centroids[j]) #build the distance matrix
        min_cluster_ids = np.argmin(dist_matrix, axis=1)
        clus = {}
        for i in range(k):
            clus[i] = []

        for i, cluster_id in enumerate(min_cluster_ids):
            clus[cluster_id].append(features[i])
        clusters=clus
        return clusters
    def __init__(self):
        pass
    def learn(self,features,k,threshold=1):
        initialcentro=self.get_centroids_random(features,k)
        converge=False
        new_centro=initialcentro
        while (not converge):
            previous_centro = new_centro
            clusters = self.clusters(features, previous_centro)

            new_centro = np.array([np.mean(clusters[key], axis=0, dtype=features.dtype) for key in sorted(clusters.keys())])

            converge = self.converge(previous_centro, new_centro,threshold)
        self.finalcluseters=self.clusters(features, previous_centro)    
    def infer(self):
        return self.finalcluseters
    