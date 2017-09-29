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

##data handling
data = pd.read_csv('irisDataSet.csv')
data = data.drop(['sepal length', 'sepal width'], axis = 1)    #Drop unwanted features
x_data = data.drop('class', axis = 1)    #separate x from data
y_data = data.drop(['petal length', 'petal width'], axis = 1)    #separate y from data

##normalize features
mms = preprocessing.MinMaxScaler()
x = x_data.values
x = mms.fit_transform(x)
x_data = pd.DataFrame(x, columns = ['petal length', 'petal width'])

##k-means
index_list = range(150)
randPoints = random.sample(index_list, 2)    #To avoid selecting the same numbers
#init array
dist_array = np.array([])
class_array = np.array([0]*150)
#for all points
for index in range(150):
    #calculate distance from first centroid to a point
    dist_array = np.append(dist_array, np.linalg.norm(x_data.values[index] - x_data.values[randPoints[0]])) 
    #get the rest distance from other centroids to a point, and find min
    for centroid in range(1, 2):
        if(np.linalg.norm(x_data.values[index] - x_data.values[centroid]) < dist_array[index]):
            dist_array[index] =  np.linalg.norm(x_data.values[index] - x_data.values[centroid])
            class_array[index] = centroid
#print(np.ndarray.tolist(class_array))
#get centroid
for centroid in range(0, 2):
    randPoints[centroid] = 