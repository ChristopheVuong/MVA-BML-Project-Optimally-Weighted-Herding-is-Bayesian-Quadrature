#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from genGM import *
from utils import *


def score(samples,kernel,gm, new_samples):
    s = 0
    for k in range(len(gm.means)):
        s += gm.weights[k]*E_Gaussian(new_samples,kernel.covariance,gm.means[k],gm.covariances[k])
    s *= 2
    n = len(samples)
    for sample in samples:
        s -= kernel.pdf(new_samples,sample)/(n+1)
    return s


def generate_herding(kernel,gm,area,samples, num_queries=10000):
    d = area.shape[0]
    new_samples = area[None, 0] + \
        (area[None, 1] - area[None, 0]) * nr.rand(num_queries, d)
    scores = score(samples, kernel, gm, new_samples)
    return new_samples[np.argmax(scores),:]

# maybe add numba here if increase  num_samples (not only numpy-like though)
def herding(num_samples, kernel, gm, area, samples=[]):
    for k in range(num_samples):
        print(k)
        samples.append(generate_herding(kernel, gm, area, samples))
    return np.array(samples)


def MMD2H_Gaussian(kernel,gm,samples):
    N = len(samples)
    
    EE = EE_Gaussian(gm,kernel)
    
    crossed_term = 0
    for sample in samples:
        for k in range(len(gm.means)):
            crossed_term += gm.weights[k]*E_Gaussian(np.array([sample]),kernel.covariance,gm.means[k],gm.covariances[k])
    
    gram_term = 0
    for sample_n in samples:
        for sample_m in samples:
            gram_term += kernel.pdf(sample_n,sample_m)
    
    return EE - 2*crossed_term/N + gram_term/N**2
