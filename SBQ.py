#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
from Herding import *
from utils import *


def update(k,gram,x,kernel,gm,samples):
    for l in range(k):
        g = kernel.pdf(np.array(samples[l]),np.array(x))
        gram[l,k] = g
        gram[k,l] = g
        
        
def scoreSBQ(samples,kernel,gm,new_samples,gram,z,k):
    g = gram.copy()
    newZ = z.copy()
    zz = np.array([newZ for l in range(len(new_samples))])
    
    for l in range(len(gm.means)):
        zz[:,k] += gm.weights[l]*E_Gaussian(new_samples,kernel.covariance,gm.means[l],gm.covariances[l])
        
    scores = []
    for l,sample in enumerate(new_samples):
        update(k,g,sample,kernel,gm,samples)
        scores.append(zz[l,:].T@np.linalg.inv(g)@zz[l,:])
    
    return scores


def scoreSBQ(samples,kernel,gm,new_samples,gram,z,k):
    """
        With inversion of partitioned matrix formula
        http://www.gaussianprocess.org/gpml/chapters/RWA.pdf p3 (201)
    """
    
    g = gram.copy()
    newZ = z.copy()
    zz = np.array([newZ for l in range(len(new_samples))])
    
    for l in range(len(gm.means)):
        zz[:,k] += gm.weights[l]*E_Gaussian(new_samples,kernel.covariance,gm.means[l],gm.covariances[l])
        
    scores = []
    K1_1 = np.linalg.inv(g)
    for l,sample in enumerate(new_samples):
        update(k,g,sample,kernel,gm,samples)
        
        if(k==0):
            K_tilde = np.linalg.inv(g)
        else:

            k_tilde = 1/(1-g[k,:k]@K1_1[:k,:k]@g[:k,k])
            K_tilde = np.eye(g.shape[0])
            
            K_tilde[:k,:k] = K1_1[:k,:k]+k_tilde*K1_1[:k,:k]@g[:k,k].reshape(-1,1)@g[k,:k].reshape(1,-1)@K1_1[:k,:k]
            K_tilde[k,:k] = -k_tilde*g[k,:k].reshape(1,-1)@K1_1[:k,:k]
            K_tilde[:k,k] = K_tilde[k,:k].T
            K_tilde[k,k] = k_tilde
        
        scores.append(zz[l,:].T@K_tilde@zz[l,:])
    
    return scores


def generate_SBQ(kernel,gm,area,samples,gram,z,k,num_queries=1000):
    d = area.shape[0]
    new_samples = area[None, 0] + \
        (area[None, 1] - area[None, 0]) * nr.rand(num_queries, d)
    
    scores = scoreSBQ(samples,kernel,gm,new_samples,gram,z,k)
    return new_samples[np.argmax(scores),:]


def SBQ(num_samples,kernel,gm,area,samples=[]):
    gram = np.eye(len(samples)+num_samples)*kernel.pdf([0,0],[0,0])
    z = np.zeros(len(samples)+num_samples)
    fillGram(gram,z,kernel,gm,samples)
    for k in range(len(samples),num_samples+len(samples)):
        print(k)
        samples.append(generate_SBQ(kernel,gm,area,samples,gram,z,k))
        update(k,gram,samples[-1],kernel,gm,samples)
        
        for l in range(len(gm.means)):
            z[k] += gm.weights[l]*E_Gaussian(np.array([samples[-1]]),kernel.covariance,gm.means[l],gm.covariances[l])
        
    return np.array(samples),gram,z


def MMD2SBQ_Gaussian(kernel,gm,samples,gram,z):
    N = len(samples)
    EE = EE_Gaussian(gm,kernel)
    return EE - z.T@np.linalg.inv(gram)@z
