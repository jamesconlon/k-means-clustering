# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:50:44 2017

@author: James
"""

import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

boundary = 100

centroids= []

clusters = 3


for i in range(clusters):
    xi = int(random.random()*boundary)
    yi = int(random.random()*boundary)                     
    centroids.append([xi,yi])
    
shape_centroids = np.reshape(centroids,(clusters,2))
print(shape_centroids)


#making random blobs

X, x = make_blobs(n_samples=100,centers = centroids,cluster_std=1)

C, c = make_circles(n_samples = 1500,shuffle = False, noise = 0.05, factor = 0.5)
print('shape',c.shape)
fig = plt.gcf()
fig.set_size_inches(12,12)

is_red = c ==0
is_blue = c ==1

plt.scatter(C[is_red,0],C[is_red,1],color = 'r')
plt.scatter(C[is_blue,0],C[is_blue,1],color = 'b')
plt.show()
temp_c = C

colormap = np.array(['black','green','cyan','yellow','orange','magenta','brown']*10)
test_clusters = 3

test_db = DBSCAN(eps = .2)
test_db.fit(C)
test_labels = test_db.labels_
test_i = test_db.core_sample_indices_
print("\n","number of clusters found: ",max(test_labels))

fig = plt.gcf()
fig.set_size_inches(12,12)


plt.scatter(C[:,0],C[:,1],color = colormap[test_labels].tolist())
plt.show()
new_c = C

print("Kmeans:")

means = KMeans(n_clusters = test_clusters,)
means.fit(C)
centers = means.cluster_centers_
kmeans_labels= means.labels_

fig = plt.gcf()
fig.set_size_inches(12,12)
plt.scatter(C[:,0],C[:,1],c = colormap[kmeans_labels])
plt.show()







