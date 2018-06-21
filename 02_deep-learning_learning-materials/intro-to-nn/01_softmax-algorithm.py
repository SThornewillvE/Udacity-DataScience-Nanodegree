# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:58:58 2018

@author: SimonThornewill
"""

import numpy as np

# Write a function that takes as an input a list of numbers, and returns
# the list of valeus given by the softmax function

def softmax(L):
    
    # Convert L to vector
    L_vec = np.asanyarray(L)
    
    # Apply exponential function
    L_vec = np.exp(L)
    
    # Take the sum
    L_sum = L_vec.sum()
    
    # Normalise based on sum
    L_vec = L_vec/L_sum
    
    return L_vec