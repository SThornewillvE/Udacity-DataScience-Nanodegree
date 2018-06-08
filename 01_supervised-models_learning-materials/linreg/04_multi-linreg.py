# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 09:58:04 2018

@author: SimonThornewill
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the data from the boston house-prices dataset 
boston_data = load_boston()
X = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
boston_model = LinearRegression()
boston_model.fit(X, y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]

prediction = boston_model.predict(sample_house)