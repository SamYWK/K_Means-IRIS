# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:25:03 2017

@author: pig84
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random

###data handling###
data = pd.read_csv('irisDataSet.csv')
data = data.drop(['sepal length', 'sepal width'], axis = 1)    #Drop unwanted features
x_data = data.drop('class', axis = 1)    #separate x from data
y_data = data.drop(['petal length', 'petal width'], axis = 1)    #separate y from data

###normalize features###
mms = preprocessing.MinMaxScaler()
x = x_data.values
x = mms.fit_transform(x)
x_data = pd.DataFrame(x, columns = ['petal length', 'petal width'])

###k-means###
def kMeans(k):
    print('\nWhen k equals to :', k)
    
    ###################avoid selecting the same points(Step 2)###################
    index_list = range(150)
    while(True):
        isSame = False
        randPoints = x_data.values[random.sample(index_list, k)]
        for i in range(k):
            for j in range(i+1, k):
                if(np.array_equal(randPoints[i], randPoints[j])):
                    print(randPoints[i], randPoints[j])
                    isSame = True
        if isSame == False:
            break
    ###################avoid selecting the same points(Step 2)###################
    
    tmpCentroid = np.array([[-1, -1]]*k)
    while (not np.array_equal(tmpCentroid, randPoints)):
        #init array
        tmpCentroid = np.copy(randPoints)
        dist_array = np.array([])
        class_array = np.array([0]*150)
        ###################divide points(Step 3)###################
        #for all points
        for index in range(150):
            #calculate distance from first centroid to a point
            dist_array = np.append(dist_array, np.linalg.norm(x_data.values[index] - randPoints[0]))
            #get the rest distance from other centroids to a point, and find min
            for centroid in range(1, k):
                if(np.linalg.norm(x_data.values[index] - randPoints[centroid]) < dist_array[index]):
                    dist_array[index] =  np.linalg.norm(x_data.values[index] - randPoints[centroid])
                    class_array[index] = centroid
        ###################divide points(Step 3)###################
        
        ###################get new centroid(Step 4)###################
        sum_x = np.zeros([k, 2])    #init sum array
        point_num = np.zeros([k, 1])    #init point_count array
        #for all centroid
        for centroid in range(0, k):
            #for all points
            for index in range(150):
                if(class_array[index] == centroid):
                    sum_x[centroid] = sum_x[centroid] + x_data.values[index]
                    point_num[centroid] += 1
            randPoints[centroid] = sum_x[centroid]/point_num[centroid]    #randPoints now become the new centroid
        ###################get new centroid(Step 4)###################
        
    #plot
    plt.axis([0, 1, 0, 1])
    for index in range(150):
        if class_array[index] == 0:
            plt.plot(x_data.values[index, 0], x_data.values[index, 1], 'r.')
        elif class_array[index] == 1:
            plt.plot(x_data.values[index, 0], x_data.values[index, 1], 'g.')
        elif class_array[index] == 2:
            plt.plot(x_data.values[index, 0], x_data.values[index, 1], 'b.')
        elif class_array[index] == 3:
            plt.plot(x_data.values[index, 0], x_data.values[index, 1], 'y.')
        elif class_array[index] == 4:
            plt.plot(x_data.values[index, 0], x_data.values[index, 1], 'k.')
    plt.show()

###call k_means###
for iteration in range(2,6):
    kMeans(iteration)