# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:53:29 2019

@author: sthornewillvonessen
"""

import numpy as np
from tqdm import *

def estimate_sigma(X, k):
    """
    Estimates Sigma using KNN to find the nearest K points. The distances to each point are then used to estimate
    a standard deviation for each point.
    
    Finally, this list of standard deviations is aggregated via the mean to give a final estimate, which is returned.
    
    :input: X - Matrix of points
    :returns: sigma_hat - an estimate of sigma
    """
    
    std = []
    
    for i in tqdm(X):
        diffs = []
        for j in X:
            # Find relevant vectors
            diff = i - j
            
            # Calculate distances
            D = (diff ** 2)
            norms = np.sqrt(D.sum())
            
            # Remember distances from this point
            diffs.append(norms)
        
        # Convert to array and find the k closest values
        diffs = np.array(diffs)
        index =  np.argpartition(diffs, k)[:k]

        std.append(diffs[index].std())

    return np.mean(std)

def Main():
    pass

if __name__ == '__main__':
    Main()