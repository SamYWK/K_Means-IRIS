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
k = 5
index_list = range(150)
randPoints = x_data.values[random.sample(index_list, k)]    #To avoid selecting the same numbers
tmpCentroid = np.array([[-1, -1], [-1, -1]])
while (tmpCentroid.all != randPoints.all):
    tmpCentroid = randPoints
    #init array
    dist_array = np.array([])
    class_array = np.array([0]*150)

    #for all points
    for index in range(150):
        #calculate distance from first centroid to a point
        dist_array = np.append(dist_array, np.linalg.norm(x_data.values[index] - randPoints[0]))
        #get the rest distance from other centroids to a point, and find min
        for centroid in range(1, k):
            if(np.linalg.norm(x_data.values[index] - randPoints[centroid]) < dist_array[index]):
                dist_array[index] =  np.linalg.norm(x_data.values[index] - randPoints[centroid])
                class_array[index] = centroid

    #get centroid
    sum_x = np.zeros([k, 2])    #init sum array
    point_num = np.zeros([k, 1])    #init point_count array
    #for all centroid
    for centroid in range(0, k):
        #for all points
        for index in range(150):
            if(class_array[index] == centroid):
                sum_x[centroid] = sum_x[centroid] + x_data.values[index]
                point_num[centroid] += 1
        if point_num.all != 0:
            randPoints[centroid] = sum_x[centroid]/point_num[centroid]    #randPoints now become the new centroid
    print(randPoints)
aw = np.array([[0], [0]])
bw = np.array([[0], [1]])
print((aw==bw).any())