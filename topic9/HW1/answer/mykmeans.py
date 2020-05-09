# -*- coding: utf-8 -*-
"""
@author: Alejandro Carderera
"""

import numpy as np
import time
import matplotlib.pyplot as plt

#Input corresponds to a bitmap image and number of clusters.
#Output corresponds to the image clustered and the cluster centers.
def mykmeans(image, clusterNum):
    print("\nRunning K-means.")
    #Number of color channels.
    dimension = 3
    #Number of bit per pixel per channel.
    bits = 8
    #Choose random cluster locations, considering its 8 bit images.
    centers = (2**bits - 1)* np.random.rand(clusterNum, dimension)
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
        centers, clusterNum = moveCluster(image, cluster, clusterNum, dimension)
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

#Assing the pixel in the image to a cluster
def assignCluster(image, centers, cluster):
    #Iterate through image pixels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #Iterate through the cluster centers.
            cluster[i,j] = 0
            d = image[i,j,:] - centers[0]
            minDim = np.dot(d, d)
            for k in range(1, centers.shape[0]):
                d = image[i,j,:] - centers[k]
                if(np.dot(d, d) < minDim):
                    minDim = np.dot(d, d)
                    cluster[i,j] = k
    return

#Move the cluster center according to assignment.
def moveCluster(image, cluster, clusterNum, dimension):
    newCenter = np.zeros((clusterNum, dimension))
    counter = np.zeros(clusterNum)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            counter[cluster[i,j]] += 1
            newCenter[cluster[i,j]] += image[i,j,:]
    #If any of the clusters does not have any pixels assigned to it, delete it.
    #Apply a reordering to the cluster assignment afterwards.
    if(np.any(counter == 0.0)):
        indexes = np.where(counter == 0)[0]
        for i in range(len(indexes)):
            clusterNum -= 1
            newCenter = np.delete(newCenter, indexes[-i - 1], 0)
            counter = np.delete(counter, indexes[-i - 1], 0)
            cluster -= (cluster >= indexes[-i - 1]).astype(np.uint8)
    for k in range(clusterNum):
        newCenter[k] = np.divide(newCenter[k], counter[k])
    return newCenter, clusterNum

#Measure the distortion function
def measureDistortion(image, cluster, centers):
    aux = 0.0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            d = image[i,j,:] - centers[cluster[i,j]]
            aux += distanceMetric(d)
    return aux

#Specifies the distance metric we will be using.
def distanceMetric(v):
    return np.linalg.norm(v, 2)
            
if __name__ == "__main__":
    import os
    
    ImageName = "your_image.jpg"
    filepath = os.path.join(os.getcwd(), "SourceImages", ImageName)
    image = importImage(filepath)
    numClusters = 8
    cluster, clusterCenters, dist, timing, clusternum = mykmeans(image, numClusters)
    
    
    #Show the original image and the clustered image.
    f, axarr = plt.subplots(1,2)
    f.suptitle('Clusters: ' + str(len(clusterCenters)))
    axarr[0].set_title('Original Image.')
    axarr[0].imshow(image)
    axarr[1].set_title('K-means Image.')
    axarr[1].imshow(cluster)
    plt.show()


