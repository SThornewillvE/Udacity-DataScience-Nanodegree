# -*- coding: utf-8 -*-

import numpy as np

def abs_error(X, y):
    """
    Calculates absolute error of a px2 matrix with its px1 predicted output
    """
    
    y_hat = np.matrix(y).transpose()
    y = X[:, 1]
    m=len(y)
    
    error = (1/m)*sum(abs(y-y_hat))

    return float(error)

def mean_error(X, y):
    """
    Calculates mean error of a px2 matrix with its px1 predicted output
    """
    
    y_hat = np.matrix(y).transpose()
    y = X[:, 1]
    m=len(y)
    
    error = (1/m)*sum(np.square(y-y_hat))

    return float(error)

# Import X matrix
X = "(2, -2), (5, 6), (-4, -4), (-7, 1), (8, 14)"

# Wrangle data to correct form
X = X.replace("(", "")
X = X.replace("),", ";")
X = X.replace(")", "")

# Convert to np.matrix
X = np.matrix(X)

# Parameters for hypothesis
theta = [2, 1.2]

# Predict y values
y = []
for i in X[:, 0]:
    y.append(theta[1]*int(i)+theta[0])
    
# Calculate Absolute error
e_1 = abs_error(X, y)
print(e_1)

# Calculate Mean error
e_2 = mean_error(X, y)
print(e_2)