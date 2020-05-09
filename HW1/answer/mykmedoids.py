# -*- coding: utf-8 -*-
"""
@author: Alejandro Carderera
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#Input corresponds to a bitmap image and number of clusters.
#Output corresponds to the image clustered and the cluster centers.
def mykmedoids(image, clusterNum):
    print("\nRunning K-medoids.")
    #Number of color channels.
    dimension = 3
    #Choose random pixels as centroids.
    centers = np.zeros((clusterNum, dimension))
    size = image.shape
    for i in range(clusterNum):
        centers[i] = image[np.random.randint(0, size[0]), np.random.randint(0, size[1]), :]
    #Will store the cluster to which each pixel corresponds.
    cluster = np.zeros((image[:,:,0].shape), dtype = np.uint8)
    #Store the value of the distortion function after we update the centers and the assignments.
    distortion = [measureDistortion(image, cluster, centers)]
    timing = [time.time()]
    clusterNumber = [clusterNum]
    print("Beginning distortion: " + str(distortion[-1]))
    print("Beginning cluster: " + str(clusterNum))
    while(len(distortion) == 1 or distortion[-2] - distortion[-1] > 0.0):
        assignCluster(image, centers, cluster)
        distortion.append(measureDistortion(image, cluster, centers))
        timing.append(time.time())
        clusterNumber.append(clusterNum)
        centers, clusterNum = moveCluster(image, cluster, centers, clusterNum, dimension)
        distortion.append(measureDistortion(image, cluster, centers))
        timing.append(time.time())
        clusterNumber.append(clusterNum)
        print("Distortion value: " + str(round(distortion[-1], 2)) + '\t' + "Distortion delta: " + str(round(distortion[-2] - distortion[-1], 2)) + '\t' + "Number of clusters: " + str(clusterNum))
    timing[:] = [t - timing[0] for t in timing]
    return cluster, centers
#    return cluster, centers, distortion, timing, clusterNumber

#Given a filepath, import the image
#Each individual channel i can be accesed with im[:,:,i]
def importImage(filepath):
    import imageio
    return imageio.imread(filepath)

#Assing the pixel in the image to a cluster, using distance generating function.
#Using np.linalg.norm we can use different distance metrics.
def assignCluster(image, centers, cluster):
    #Iterate through image pixels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #Iterate through the cluster centers.
            cluster[i,j] = 0
            d = image[i,j,:] - centers[0]
            minDim = distanceMetric(d)
            for k in range(1, centers.shape[0]):
                d = image[i,j,:] - centers[k]
                if(distanceMetric(d) < minDim):
                    minDim = distanceMetric(d)
                    cluster[i,j] = k
    return

#Specifies the distance metric we will be using.
def distanceMetric(v):
    return np.linalg.norm(v, 1)

#Move the cluster center according to assignment.
#Needs to be consistent with the distance function.
#Partitioning Around Medoids (PAM) algorithm.
def moveCluster(image, cluster, center, clusterNum, dimension):
    newCenter = np.zeros((clusterNum, dimension))
    counter = np.zeros(clusterNum)
    #For each cluster choose a new representative that decreases discrepancy
    for k in range(clusterNum):
        points = image[(cluster == k), 0:dimension].astype(np.float)
        counter[k] = len(points)
        #Choose median for each coordinate.
        import statistics
        optCenter = np.array([statistics.median(points[:,0]), statistics.median(points[:,1]), statistics.median(points[:,2])])
        #Calculate distance from this point to all others
        distances = np.abs(points - optCenter).sum(axis = 1)
        sortedIndexes = np.argsort(distances)
        newCenter[k] = points[sortedIndexes[0]]
        #Search among the sorted distances:
        minDist = np.abs(points - newCenter[k]).sum()
        for i in range(1, min(len(points),100)):
            dist = np.abs(points - points[sortedIndexes[i]]).sum()
            if(dist < minDist):
                minDist = dist
                newCenter[k] = points[sortedIndexes[i]]
        #If cluster center is worse than the initial one, return the initial one.
        dist = np.abs(points - center[k]).sum()
        if(dist < minDist):
            newCenter[k] = center[k]
    return newCenter, clusterNum

#Move the cluster center according to assignment.
#Needs to be consistent with the distance function.
#Partitioning Around Medoids (PAM) algorithm.
def moveClusterBackup(image, cluster, center, clusterNum, dimension):
    newCenter = np.zeros((clusterNum, dimension))
    counter = np.zeros(clusterNum)
    #For each cluster choose a new representative that decreases discrepancy
    for k in range(clusterNum):
        points = image[(cluster == k), 0:dimension].astype(np.float)
        counter[k] = len(points)
        oldDist = np.abs(points - center[k]).sum()
        for l in range(0, len(points)):
            dist = np.abs(points - points[l]).sum()
            if(dist < oldDist):
                oldDist = dist
                newCenter[k] = points[l]
    return newCenter, clusterNum

#Measure the distortion function
def measureDistortion(image, cluster, centers):
    aux = 0.0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            d = image[i,j,:] - centers[cluster[i,j]]
            aux += distanceMetric(d)
    return aux
            
if __name__ == "__main__":
    import os
    
    filepath = os.path.join(os.getcwd(), "beach.bmp")
    image = importImage(filepath)
    numClusters = 3
    cluster, clusterCenters = mykmedoids(image, numClusters)
    
    
    #Show the original image and the clustered image.
    f, axarr = plt.subplots(1,2)
    f.suptitle('Clusters: ' + str(len(clusterCenters)))
    axarr[0].set_title('Original Image.')
    axarr[0].imshow(image)
    axarr[1].set_title('K-means Image.')
    axarr[1].imshow(cluster)
    plt.show()


