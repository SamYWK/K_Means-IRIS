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

#data handling
data = pd.read_csv('irisDataSet.csv')
data = data.drop(['sepal length', 'sepal width'], axis = 1)    #Drop unwanted features
x_data = data.drop('class', axis = 1)    #separate x from data
y_data = data.drop(['petal length', 'petal width'], axis = 1)    #separate y from data
#print(x_data)
#print(y_data)
#normalize features
mms = preprocessing.MinMaxScaler()
x = x_data.values   
x = mms.fit_transform(x)
x_data = pd.DataFrame(x, columns = ['petal length', 'petal width'])
plt.scatter(x_data.values[:, 0], x_data.values[:, 1])
plt.show()

#k-means
index_list = range(150)
randPoints = random.sample(index_list, 2)    #To avoid selecting the same numbers
