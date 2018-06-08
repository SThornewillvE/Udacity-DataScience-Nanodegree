# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:50:40 2018

@author: SimonThornewill
"""

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('lib/01_decision-trees_data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# Create model
data_model = DecisionTreeClassifier()

# Fit the Model
data_model.fit(X, y)

# Make prediction
y_pred = data_model.predict(X)

# Calculate accuracy
acc = accuracy_score(y, y_pred)