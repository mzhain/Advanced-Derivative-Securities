import random
from scipy.stats import norm

def RandN():
    return norm.ppf(random.random())

def cumulative_standard_normal(d):
    return norm.cdf(d)
