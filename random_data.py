# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 01:39:38 2017

@author: James
"""

#necessary libraries
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


##########
# (1) Customizing input variables 
# feel free to change any of the following:
n_clusters = 5 
n_samples = 300
cluster_std = 20
method = 'random' # can be changed to 'k-means++'
###########

boundary = 100 #rough scope of the plot is boundary X boundary. This can be changed

'''
Don't change anything below here
'''

# (2) randomizing the location of the clusters
centroids= []
for i in range(n_clusters):
    xi = int(random.random()*boundary)
    yi = int(random.random()*boundary)                     
    centroids.append([xi,yi])
    
generated_centroids = np.reshape(centroids,(n_clusters,2))
print('generated centroids\n',generated_centroids)


# (3) making random blobs around centroids
X, x = make_blobs(n_samples=n_samples,centers = centroids,cluster_std=cluster_std)
X_orig = np.column_stack((X,x))

fig = plt.gcf()
fig.set_size_inches(12,12)

plt.scatter(X[:,0],X[:,1],color = 'k')
plt.scatter(generated_centroids[:,0],generated_centroids[:,1],marker='D', color='b')

colormap = np.array(['black','green','cyan','yellow','orange','magenta','brown']*10)

# (4) Using k-Means to fit the randomized data
test_means = KMeans(n_clusters = n_clusters,init=method) 
test_means.fit(X)
test_centers = test_means.cluster_centers_
test_labels = test_means.labels_


# (5) Plotting results (and making sure the estimated centroids are in the correct order)

def getDist(x1,y1,x2,y2):
    x = x1-x2
    y= y1-y2
    return((x**2+y**2)**.5)

ref_array = [-1]*5
for i in range(n_clusters):
    dist_array = [1000]*20

    temp_set = centroids[i]
    temp_x = temp_set[0]
    temp_y = temp_set[1]
    temp_index = 0
    temp_dist_array = []
    for j in range(n_clusters):
       test_set = test_centers[j]
       test_x = test_set[0]
       test_y = test_set[1]
       temp_dist_array.append(getDist(temp_x,temp_y,test_x,test_y))
    ref_array[i] = temp_dist_array.index(min(temp_dist_array))

location_match = []
for i in range(n_clusters):
    location_match.extend(test_centers[ref_array[i]])
    
location_match = np.reshape(location_match,(n_clusters,2))
final_test_centers = location_match
print('estimated centroids\n',final_test_centers.round(1))


plt.scatter(X[:,0],X[:,1],c = colormap[test_labels])
plt.scatter(final_test_centers[:,0],final_test_centers[:,1],marker = 'X', color = 'r')
plt.show()
