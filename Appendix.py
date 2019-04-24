import random
from scipy.stats import norm

def RandN():
    return norm.ppf(random.random())

