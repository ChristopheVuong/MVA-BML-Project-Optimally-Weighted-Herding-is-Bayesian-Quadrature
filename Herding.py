#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from utils import E_Gaussian, EE_Gaussian, progress


def score(samples,kernel,gm, new_samples):
    """
        Compute the score E[k(x, x')] - 1 / (n+1) ones.T K_x

        Inputs:
        - samples: the existing samples in the herd
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture object
        - new_samples: random samples (select one)  
    """
    s = E_Gaussian(new_samples, gm, kernel)
    s *= 2
    n = len(samples)
    for sample in samples:
        s -= kernel.pdf(new_samples,sample)/(n+1)
    return s


def generate_herding(kernel,gm,area,samples, num_queries=10000):
    """
        Choose among random samples the one with maximum score
        Equivalent to minimizing MMD

        Inputs:
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - area: the bounds where to herding
        - samples: the existing samples in the herd
        - num_queries: the number of random points to draw for a new sample 
    """
    d = area.shape[0]
    new_samples = area[None, 0] + \
        (area[None, 1] - area[None, 0]) * np.random.rand(num_queries, d)
    scores = score(samples, kernel, gm, new_samples)
    return new_samples[np.argmax(scores),:]


def herding(num_samples, kernel, gm, area, samples=[]):
    """
        Kernel herding: at each step choose points that maximizes the score

        Inputs:
        - num_samples: the number of herding samples 
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - area: the bounds where to do herding
        - samples: the existing samples in the herd
    """
    for k in range(num_samples):
        progress(k, num_samples, 'Kernel Herding')
        samples.append(generate_herding(kernel, gm, area, samples))
    return np.array(samples)



def MMD2H_Gaussian(kernel,gm,samples):
    """
        Compute the maximum mean descrepancy between the actual distribution
        and the uniform distribution over the selected samples

        Inputs:
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - samples: all the herding samples
    """
    N = len(samples)
    
    EE = EE_Gaussian(gm,kernel)
    
    crossed_term = 0
    for sample in samples:
        crossed_term += E_Gaussian(np.array([sample]), gm, kernel)

    gram_term = 0
    for sample_n in samples:
        for sample_m in samples:
            gram_term += kernel.pdf(sample_n,sample_m)

    return EE - 2*crossed_term/N + gram_term/N**2
