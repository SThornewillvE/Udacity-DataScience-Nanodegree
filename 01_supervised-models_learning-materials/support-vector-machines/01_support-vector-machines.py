# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:07:17 2018

@author: SimonThornewill
"""

# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def data_split(data):
    X = data[:,0:2] 
    return X

# Read the data.
data = np.asarray(pd.read_csv('lib/01_support-vector-machines_data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# Plot Data
X_0 = data_split(data[data[:, 2]==0])
X_1 = data[data[:, 2]==1]

plt.scatter(X_0[:, 0], X_0[:, 1], c="red")
plt.scatter(X_1[:, 0], X_1[:, 1], c="blue")
plt.show()

# Find the right parameters for this model to achieve 100% accuracy on the dataset.
param_grid = {'gamma': np.arange(1,40)}
model = GridSearchCV(SVC(), param_grid)

# Fit the model.
model.fit(X,y)

# Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)