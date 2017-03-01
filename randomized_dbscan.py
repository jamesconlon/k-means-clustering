# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:56:06 2017

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
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt




boundary = 100 #initializes so plot is 100 x 100



total_clusters = 6
all_centers = []

moon_count, circle_count = 1,0 #this can be edited, but works best when one is 1 and the other is 0

total_samples = 0 #don't change this
'''#can use this if you want circles or moons to be randomly thrown in given their probabilities


blob_prob = 0.8
circle_prob = 0.1
moon_prob = 0.1

for i in range(total_clusters):
    rnd = random.random()
    if (rnd <moon_prob):
        moon_count = 1 #probably don't want more than one moon or circle in the data. DBSCAN will almost certainly return as [-1] (noise)
    elif(rnd <(moon_prob+circle_prob)):
        circle_count = 1
'''


blob_count = total_clusters - (circle_count+moon_count)

print(blob_count,circle_count,moon_count)



blob_samples = 500
if(blob_count>0):
    
    blob_centers= [] #empty list for blob centroids to be appended to

    blob_clusters = blob_count #should be a parameter later --> blob_count
    blob_mean = 10

    for i in range(blob_clusters):
        xi = int(random.random()*boundary)
        yi = int(random.random()*boundary)                     
        blob_centers.append([xi,yi])
        all_centers.append([xi,yi])
    
    blob_centers = np.reshape(blob_centers,(blob_clusters,2))

    blob_sd = np.random.normal(loc = blob_mean, scale = 2.5,size = 5)
    print('sd',blob_sd)

    X, x_index = make_blobs(n_samples=blob_samples,centers = blob_centers,shuffle = False, cluster_std=blob_sd)
    total_samples = total_samples + blob_samples



circle_samples = 500

if(circle_count>0):

    rnd_x = (random.random()*2-1)*5+50
    rnd_y = (random.random()*2-1)*5+50
    all_centers.append([rnd_x,rnd_y])
    all_centers.append([rnd_x,rnd_y]) #twice since two circles are made
    #circle_samples = 500
    circle_offset = blob_count
    C, c = make_circles(n_samples = circle_samples,shuffle = False, noise = 0.025, factor = 0.5)
    total_samples = total_samples + circle_samples
    c = c+circle_offset
    C = C*20
    C[:,0] = C[:,0]+rnd_x
    C[:,1] = C[:,1]+rnd_y

moon_samples = 500
if(moon_count>0):
    rnd_x = (random.random()*2-1)*30+50
    rnd_y = (random.random()*2-1)*30+50
    all_centers.append([rnd_x,rnd_y])
    all_centers.append([rnd_x,rnd_y]) #twice since two moons are made
    #moon_samples = 500
    moon_offset = blob_count+2*circle_count
    M,m = (make_moons(n_samples = moon_samples,shuffle = False, noise = 0.025))
    total_samples = total_samples + moon_samples
    m = m+moon_offset
    M = M*20
    M[:,0] = M[:,0]+rnd_x
    M[:,1] = M[:,1]+rnd_y


blob_silhouette = silhouette_score(X, x_index)


reference_stack = np.column_stack((X,x_index))



x_stack = np.column_stack((X,x_index))
c_stack = np.column_stack((C,c))
m_stack = np.column_stack((M,m))

complete_stack = x_stack

if(circle_count>0):
    complete_stack = np.append(x_stack,c_stack)

if(moon_count>0):
    complete_stack = np.append(complete_stack, m_stack)
    
print(complete_stack)

complete_stack = np.reshape(complete_stack,(total_samples,3))
print('new',complete_stack)


fig = plt.gcf()
fig.set_size_inches(12,12)


plt.scatter(X[:,0],X[:,1],color = 'k')
plt.scatter(blob_centers[:,0],blob_centers[:,1],marker='D', color='b') #only plots centroids

           
if(circle_count>0):
    
    circle_0 = (c ==0 +circle_offset)
    circle_1 = (c ==1 +circle_offset)
    plt.scatter(C[circle_0,0],C[circle_0,1],color = 'k')
    plt.scatter(C[circle_1,0],C[circle_1,1],color = 'k')    
           
if(moon_count > 0):
          
    moon_0 = (m ==0+moon_offset)
    moon_1 = (m ==1+moon_offset)
    plt.scatter(M[moon_0,0],M[moon_0,1],color = 'k')
    plt.scatter(M[moon_1,0],M[moon_1,1],color = 'k')   
           
def getDist(x1,y1,x2,y2):
    x = x1-x2
    y= y1-y2
    return((x**2+y**2)**.5)
     

def getCost(stack,centers):
    i_range = range(len(stack))
    
    cost_list = []
    for i in i_range:
        temp = stack[i]
        temp_x = temp[0]
        temp_y = temp[1]
        temp_index = int(temp[2])
    
        center = centers[temp_index]
        center_x = center[0]
        center_y = center[1]
    
        dist = getDist(temp_x,temp_y,center_x,center_y)
        cost_list.append(dist)
    cost = sum(cost_list)
    #print('max cost',max(cost_list))
    return(cost)

#blob cost is the minimum cost (where every point is in the correct cluster)           
blob_cost = getCost(x_stack,blob_centers)
sum_cost = blob_cost
if(circle_count>0):
    circle_cost = getCost(c_stack,all_centers)
    sum_cost = sum_cost + circle_cost
    
if(moon_count>0):
    moon_cost = getCost(m_stack,all_centers)
    sum_cost = sum_cost + moon_cost

randomization_cost = getCost(complete_stack,all_centers)
average_randomization_cost = randomization_cost / total_samples

average_blob_cost = blob_cost / blob_samples

#print('all cost',randomization_cost)


plt.show()
fig = plt.gcf()
fig.set_size_inches(12,12)


'''
Attempting to find best fit for the randomized data
'''           

#print(blob_centers[reference_stack[0][2]]) #this gets the corresponding centroid tuple (xi, yi) for point 0

#using k-means

test_data = np.column_stack((complete_stack[:,0],complete_stack[:,1])) #points for all of the samples


colormap = np.array(['black','green','cyan','yellow','orange','magenta','brown']*10)
test_clusters = 5 #should iterate through this value

k_means = KMeans(n_clusters = test_clusters,init='random')#init = ''k-means++'
k_means.fit(test_data)

k_means_centers = k_means.cluster_centers_
k_means_labels = k_means.labels_



plt.scatter(test_data[:,0],test_data[:,1],color = colormap[k_means_labels].tolist())

test_data_stack = np.column_stack((test_data,k_means_labels))

k_means_cost = getCost(test_data_stack,k_means_centers)
average_k_means_cost = k_means_cost / total_samples


print('avg blob, kmeans', average_blob_cost,average_k_means_cost)

#print('tc:',test_center)
#print(test_labels) #0-4
plt.show()
fig = plt.gcf()
fig.set_size_inches(12,12)

  
#DBSCAN

db_test_data = np.column_stack((complete_stack[:,0],complete_stack[:,1]))


dbscan = DBSCAN(eps = 9, min_samples = 50)
dbscan.fit(db_test_data)
dbscan_labels = dbscan.labels_
print(dbscan_labels)
#c = dbscan.components_ #this is the same as X
print('db max',max(dbscan_labels))

plt.scatter(db_test_data[:,0],db_test_data[:,1],color = colormap[dbscan_labels].tolist())
plt.show()
print(colormap[-1],'corresponds to noise. DBSCAN assigns these as [-1]')


'''#this was used to loop through and find the best eps value. You can uncomment this block if you are stuck
max_label = 0
label_index = 0
for i in range(1000):
    eps = (i+1)/10
    dbscan = DBSCAN(eps = eps, min_samples = 50)
    dbscan.fit(db_test_data)
    dbscan_labels = dbscan.labels_
    if (max(dbscan_labels)>max_label):
        max_label = max(dbscan_labels)
        label_index = i
        
print(max_label,label_index,eps)
'''







           