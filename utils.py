#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from genGM import *


def multivariate_gaussian(pos, mu, Sigma):
    """
        Return the multivariate Gaussian distribution on array pos.

        Input:
        - pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        - mu: expectation of a gaussian
        - sigma: standard deviation of a gaussian
        
        Source: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


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


def E_Gaussian(x,sigma0,mu,sigma):
    """
        Inputs:
        - sigma0: variance of kernel
        - mu,sigma: parameters of gaussian law
    """
    inv_sigma0 = np.linalg.inv(sigma0)
    inv_sigma = np.linalg.inv(sigma)
    inv_new_sigma = inv_sigma0+inv_sigma
    new_sigma = np.linalg.inv(inv_new_sigma)
    
    new_mu = x@inv_sigma0 + (inv_sigma@mu)[None]
    # take the diagonal with np.einsum (parallelization)
    exp = np.exp(-(1/2)*(np.einsum('ij,jk,ki->i', x, inv_sigma0, x.T) +
                         (mu.T@inv_sigma@mu)-np.einsum('ij,jk,ki->i', new_mu, new_sigma, new_mu.T)))
    
    return np.sqrt(np.linalg.det(new_sigma)/(np.linalg.det(sigma0)*np.linalg.det(sigma)))*exp/(2*np.pi)


def EE_Gaussian(distrib, kernel):
    """
        Compute the prior variance of BMC when the input distribution is a distture of
        Gaussians, and the kernel is a Gaussian.
        Only for Gaussians for now !!! 
    """
    K = distrib.means.shape[0]
    prior_variance = 0
    for k in range(K):
        for j in range(K):
            cur_covariance = kernel.covariance + distrib.covariances[k]  + distrib.covariances[j] 
            prior_variance += distrib.weights[k] * distrib.weights[j] * multivariate_normal.pdf(distrib.means[k], distrib.means[j], cur_covariance)
    return prior_variance


def fillGram(gram,z,kernel,gm,samples):
    for i in range(len(samples)):
        for j in range(len(samples)):
            gram[i,j] = kernel.pdf(np.array(samples[i]),np.array(samples[j]))
            gram[j,i] = gram[i,j]

        for l in range(len(gm.means)):
            z[i] += gm.weights[l]*E_Gaussian(np.array([samples[i]]),kernel.covariance,gm.means[l],gm.covariances[l]) 
