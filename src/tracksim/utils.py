"""
Contains utility function used in the main tracksim module as well as other
useful functions.
"""

import os
import shutil
import numpy as np
from tqdm import tqdm

def moving_average(a,n):
    ma = np.cumsum(a, dtype=float)
    ma[n:] = ma[n:] - ma[:-n]
    
    return ma[n-1:]/n

def exp_average(a, alpha):
    
    ma = np.zeros(len(a))
    ma[0] = a[0]    
    for i in range(1, len(a)):
        ma[i] = alpha*a[i]+ (1-alpha)*ma[i-1]
    
    return ma

