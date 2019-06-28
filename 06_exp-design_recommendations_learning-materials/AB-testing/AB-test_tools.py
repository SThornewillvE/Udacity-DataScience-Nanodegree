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


def simulate_Pvalue(p, n_obs, n_occur, n_trials=5_000):
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


def simulate_Pvalue_test(p, n_control, n_exp, p_control, p_exp, n_trials=5_000):
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


def calculate_power(p_null, p_alt, n, alpha=0.05):
    """
    Compute power of detecting difference in two populations with different proportion parameters with
    given alpha rate.
    
    :Input:
        :p_null: Rate of success under the null hypothesis
        :p_alt: Desired rate of success under the alternative hypothesis
        :n: Sample size
        :alpha: Desired significance level (i.e. Type 1 Error Rate)
    :Returns:
        :power: 1 - beta
    """

    se_null = np.sqrt((p_null*(1-p_null) + p_null*(1-p_null)) / n)
    null_dist = stats.norm(loc = 0, scale = se_null)
    p_crit = null_dist.ppf(1 - alpha)

    se_alt  = np.sqrt((p_null*(1-p_null) + p_alt*(1-p_alt)) / n)
    alt_dist = stats.norm(loc = p_alt - p_null, scale = se_alt)
    beta = alt_dist.cdf(p_crit)

    power = 1 - beta
    
    return power


def find_experiment_size(p_null, p_alt, alpha = .05, beta = .20):
    """
    Calculate min number of samples requried to achieve a desired power level for a given effect size.
    
    :Input:
        :p_null: Rate of success under the null hypothesis
        :p_alt: Desired rate of success under the alternative hypothesis
        :alpha: Desired significance level (i.e. Type 1 Error Rate)
        :beta: Desired 1 - power (i.e. Type 2 Error Rate)
    :Returns:
        :n: Sample size
    """

    # Get necessary z-scores and standard deviations (@ 1 obs per group)
    z_null = stats.norm.ppf(1 - alpha)
    z_alt  = stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
    
    # Compute and return minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
    
    return np.ceil(n)


def Main():
    None


if __name__ == '__main__':
    Main()