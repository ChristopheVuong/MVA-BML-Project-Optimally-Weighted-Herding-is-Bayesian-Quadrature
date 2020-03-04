#!/usr/bin/env python
# coding: utf-8

import numpy as np
from utils import E_Gaussian, EE_Gaussian, fillGram, progress


def update(k,gram,x,kernel,gm,samples):
    """
        Update Gram matrix with a new sample

        Inputs:
        - k: the index for the future sample 
        - kernel: a Gaussian kernel
        - x: a potential sample
        - area: the bounds for the points
        - samples: the existing samples in the herd
    """
    for l in range(k):
        g = kernel.pdf(np.array(samples[l]),np.array(x))
        gram[l,k] = g
        gram[k,l] = g
        
        


def scoreSBQ(samples,kernel,gm,new_samples,gram,z,k):
    """
        With inversion of partitioned matrix formula
        http://www.gaussianprocess.org/gpml/chapters/RWA.pdf p3 (201)
        Equivalent to fast update 
        Next step: keep Cholesky update only:
        http://www.cs.ubbcluj.ro/~csatol/SOGP/thesis/Iterative_computation.html

        Inputs:
        - samples: the existing samples 
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - new_samples: the random points
        - gram: the gram matrix associated to the Gaussian kernel
        - z: discrete expectation E[k(x, x_n)]
    """
    
    g = gram.copy()
    newZ = z.copy()
    zz = np.array([newZ for l in range(len(new_samples))])
    
    zz[:,k] = E_Gaussian(new_samples, gm, kernel)
        
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
    """
        Choose among random samples the one with maximum score

        Inputs:
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - area: the bounds where to herding
        - samples: the existing samples in the herd
        - gram: the Gram matrix at iteration
        - z: 
        - num_queries: the number of random points to draw for a new sample 
    """
    d = area.shape[0]
    new_samples = area[None, 0] + \
        (area[None, 1] - area[None, 0]) * np.random.rand(num_queries, d)
    
    scores = scoreSBQ(samples,kernel,gm,new_samples,gram,z,k)
    return new_samples[np.argmax(scores),:]


def SBQ(num_samples,kernel,gm,area,samples=[]):
    """
        Sequential Bayesian quadrature: select points that
        maximizes the score

        Inputs:
        - kernel: a Gaussian kernel
        - gm: a Gaussian mixture
        - area: the bounds where to herding
        - samples: the existing samples in the herd
        - num_queries: the number of random points to draw for a new sample 
    """

    n = len(samples)
    gram = np.eye(n+num_samples)*kernel.pdf([0,0],[0,0])
    z = np.zeros(n+num_samples)
    fillGram(gram,z,kernel,gm,samples)
    for k in range(n,num_samples+len(samples)):
        progress(k - n, num_samples, status='Bayesian Quadrature')
        samples.append(generate_SBQ(kernel,gm,area,samples,gram,z,k))
        update(k,gram,samples[-1],kernel,gm,samples)
        
        z[k] = E_Gaussian(np.array([samples[-1]]), gm, kernel)
        
    return np.array(samples),gram,z



def MMD2SBQ_Gaussian(kernel,gm,samples,gram,z):
    EE = EE_Gaussian(gm,kernel)
    return EE - z.T@np.linalg.inv(gram)@z