#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GaussianMixture():
    def __init__(self, p, mus, sigmas):
        self.means = mus
        self.covariances = sigmas
        self.weights = p
        self.rvs = [multivariate_normal(mus[k], sigmas[k]) for k in range(len(self.weights))]

    def pdf(self, X):
        return np.sum([self.weights[i]*self.rvs[i].pdf(X) for i in range(len(self.weights))], axis=0)


class GaussianKernel():
    def __init__(self,sigma):
        self.covariance = sigma
    
    def pdf(self,x,x2):
        return multivariate_normal(x2,self.covariance).pdf(x)



def E_Gaussian(x, mix, kernel):
    """
        Inputs:
        - x the samples
        - mix: the Gaussian mixture object
        - kernel: the kernel object
    """
    # inv_sigma0 = np.linalg.inv(sigma0)
    # inv_sigma = np.linalg.inv(sigma)
    # inv_new_sigma = inv_sigma0+inv_sigma
    # new_sigma = np.linalg.inv(inv_new_sigma)
    
    # new_mu = x@inv_sigma0 + (inv_sigma@mu)[None]
    # # take the diagonal with np.einsum (parallelization)
    # exp = np.exp(-(1/2)*(np.einsum('ij,jk,ki->i', x, inv_sigma0, x.T) +
    #                      (mu.T@inv_sigma@mu)-np.einsum('ij,jk,ki->i', new_mu, new_sigma, new_mu.T)))
    
    # return np.sqrt(np.linalg.det(new_sigma)/(np.linalg.det(sigma0)*np.linalg.det(sigma)))*exp/(2*np.pi)
    K = len(mix.means)
    E = 0
    for k in range(K):
        E += mix.weights[k] * multivariate_normal.pdf(x, mix.means[k], mix.covariances[k] + kernel.covariance)
    return E


def EE_Gaussian(distrib, kernel):
    """
        Compute the prior variance of BMC when the input distribution is a mixture of
        Gaussians, and the kernel is a Gaussian.

        Inputs:
        - distrib: a mixture of Gaussians
        - kernel: a Gaussian kernel 
    """
    K = len(distrib.means)
    prior_variance = 0
    for k in range(K):
        for j in range(K):
            cur_covariance = kernel.covariance + distrib.covariances[k]  + distrib.covariances[j] 
            prior_variance += distrib.weights[k] * distrib.weights[j] * multivariate_normal.pdf(distrib.means[k], distrib.means[j], cur_covariance)
    return prior_variance


def fillGram(gram,z,kernel,gm,samples):
    """
        Function to complete the Gram matrix and z
        Used for the computations of BQ weights, MMD and errors

        Inputs:
        - gram: a matrix to complete thanks to
        - z : E[k(x_n,x)]_{n=1,...,num_samples}
        - kernel: a Gaussian kernel
    """
    for i in range(len(samples)):
        for j in range(len(samples)):
            gram[i,j] = kernel.pdf(np.array(samples[i]),np.array(samples[j]))
            gram[j,i] = gram[i,j]

        z[i] += E_Gaussian(np.array([samples[i]]), gm, kernel) 



# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev

def progress(count, total, status=''):
    # percents = round(100.0 * count / float(total), 1)
    bar = '=' * (count+1) + '-' * (total - count -1)

    sys.stdout.write('[%s] %s / %s ... %s\r' % (bar, count+1, total, status))
    sys.stdout.flush()