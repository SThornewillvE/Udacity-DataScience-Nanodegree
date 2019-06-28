# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:06:52 2019

@author: sthornewillvonessen
"""

import numpy as np
import scipy.stats as stats


def calculate_Pvalue(p, n_obs, n_occur, tails):
    """
    Calculates Z-score and P-value given a null hypothesis, number of observations
    and number of occurances of a target variable.
    
    Performs a two-tailed test.
    
    This test assumes data is binomially distributed.
    
    :Inputs:
        :p: Null hypothesis probability
        :n_obs: Number of total observations
        :n_occur: Number of times outcome occurs in observations
        :tails: Integer (1 or 2), Decides whether one or two tailed test respectively.
    :Returns:
        :z_score: Z-score of event
        :p_value: Probability of data given null hypothesis
    """
    
    if tails not in [1, 2]:
        print("Test must have one or two tails, input as an integer")
        return ValueError
    
    # Calculate Standard Deviation
    sd = np.sqrt(p * (1-p) * n_obs)
    
    # Calculate Z-score, normalised difference
    z_score = ((n_occur + 0.5) - p * n_obs) / sd
    
    # Find CDF of z_score
    p_value = tails * stats.norm.cdf(z_score)
    
    return z_score, p_value


def simulate_Pvalue(p, n_obs, n_occur, n_trials=200_000):
    """
    Simulates Z-score and P-value given a null hypothesis, number of observations
    and number of occurances of a target variable.
    
    This test assumes data is binomially distributed.
    
    :Inputs:
        :p: Null hypothesis probability
        :n_obs: Number of total observations
        :n_occur: Number of times outcome occurs in observations
        :n_trials: Number of simulations
    :Returns:
        :p_value: Probability of data given null hypothesis
    """

    samples = np.random.binomial(n_obs, p, n_trials)

    p_value = np.logical_or(samples <= n_occur, samples >= (n_obs - n_occur)).mean()
    
    return p_value


def calculate_Pvalue_test(p, n_control, n_exp, p_control, p_exp):
    """
    Calculates the Z-score and P-values for two outcomes in an experiment.
    
    This test assumes data is binomially distributed.
    
    This test is also a one-sideded test.
    
    :Inputs:
        :p: Null hypothesis probability
        :n_control: Number of control observations
        :n_exp: Number of experimental observations
        :p_control: Rate of success for control variavble
        :p_exp: Rate of success for experimental variavble
    :Returns:
        :p_value: Probability of data given null hypothesis
    """

    # compute standard error, z-score, and p-value
    se_p = np.sqrt(p * (1-p) * (1/n_control + 1/n_exp))

    z_score = (p_exp - p_control) / se_p

    p_value = 1-stats.norm.cdf(z_score)

    return z_score, p_value


def simulate_Pvalue_test(p, n_control, n_exp, p_control, p_exp, n_trials=200_000):
    """
    Calculates the Z-score and P-values for two outcomes in an experiment.
    
    This test assumes data is binomially distributed.
    
    This test is also a one-sideded test.
    
    :Inputs:
        :p: Null hypothesis probability
        :n_control: Number of control observations
        :n_exp: Number of experimental observations
        :p_control: Rate of success for control variable
        :p_exp: Rate of success for experimental variable
        :n_trials: Number of simulations
    :Returns:
        :p_value: Probability of data given null hypothesis
    """

    ctrl = np.random.binomial(n_control, p, n_trials)
    exp = np.random.binomial(n_exp, p, n_trials)
    samples = exp / n_exp - ctrl / n_control

    p_value = (samples >= (p_exp - p_control)).mean()

    return p_value


def Main():
    None


if __name__ == '__main__':
    Main()