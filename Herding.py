#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from genGM import *
# import numba
#import seaborn as sns


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
    # print(new_samples)
    scores = score(samples, kernel, gm, new_samples)
    return new_samples[np.argmax(scores),:]

# maybe add numba here if increase  num_samples (not only numpy-like though)
def herding(num_samples, kernel, gm, area, samples=[]):
    for k in range(num_samples):
        print(k)
        samples.append(generate_herding(kernel, gm, area, samples))
    return np.array(samples)


def EE_Gaussian(kernel,gm):
       
    output = 0
    sigma = kernel.covariance
    inv_sigma = np.linalg.inv(sigma)
        
    for k in range(len(gm.means)):
        sigma_k = gm.covariances[k]
        mu_k = gm.means[k]
        
        inv_sigma_k = np.linalg.inv(sigma_k)
        det_sigma_k = np.linalg.det(sigma_k)
        
        Q_k = np.linalg.inv(inv_sigma+inv_sigma_k)
        inv_S_k = inv_sigma+inv_sigma@Q_k@inv_sigma
        S_k = np.linalg.inv(inv_sigma+inv_sigma@Q_k@inv_sigma)
        
        det_Q_k = np.linalg.det(Q_k)
        det_S_k = np.linalg.det(S_k)
        
        a_k = -S_k@inv_sigma@Q_k@inv_sigma_k@mu_k
        
        exp = np.exp(-(mu_k.T@(inv_sigma_k-inv_sigma_k@Q_k@inv_sigma_k)@mu_k-a_k.T@inv_S_k@a_k)/2)

        output += gm.weights[k]*np.sqrt(det_Q_k*det_S_k/det_sigma_k)*exp*E_Gaussian(a_k,sigma,mu_k,sigma_k)
    return output/np.sqrt(np.linalg.det(sigma))


def MMD2(kernel,gm,samples):
    ## To check
    N = len(samples)
    
    EE = EE_Gaussian(kernel,gm)
    
    crossed_term = 0
    for sample in samples:
        for k in range(len(gm.means)):
            crossed_term += E_Gaussian(sample,kernel.covariance,gm.means[k],gm.covariances[k])
    
    gram_term = 0
    for sample_n in samples:
        for sample_m in samples:
            gram_term += kernel.pdf(sample_n,sample_m)
    
    print(EE,crossed_term,gram_term)
    
    return EE - crossed_term/N + gram_term/N**2
