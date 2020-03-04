#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from Herding import MMD2H_Gaussian
from utils import fillGram, progress
from SBQ import MMD2SBQ_Gaussian



def plotContour(gm,samplesH=None,samplesSBQ=None):
    stdsig = 2
    xx, yy = np.mgrid[np.min(
        gm.means[:, 0])-stdsig:np.max(gm.means[:, 0])+stdsig:.1, np.min(gm.means[:, 1])-stdsig:np.max(gm.means[:, 1])+stdsig:.1]
    # xx, yy = np.mgrid[-4:6:.1, -4:6:.1]
    pos = np.empty(xx.shape + (2,))
    pos[:,:,0] = xx; pos[:,:,1] = yy

    f = gm.pdf(pos)
    plt.figure(figsize=(10, 7))
    plt.contour(xx, yy, f, 20)
    # plt.scatter(*mus.T, facecolors='none', edgecolors='r')
    if samplesSBQ is not None:
        plt.scatter(*samplesSBQ.T,c='black',marker='o',label='SBQ Samples')
    if samplesH is not None:
        plt.scatter(*samplesH.T,c='red',marker='x',label='Herding Samples')
    if samplesH is not None or samplesSBQ is not None:
        plt.legend()
    plt.savefig('figures/samples')
    plt.show()


def plotSBQweights(samples, gm, kernel):
    n = len(samples)
    gram = np.zeros((n,n))
    z = np.zeros((n))
    fillGram(gram,z,kernel,gm,samples)          # BQ weights for kernel herding
    weights = z@np.linalg.inv(gram)
    
    plt.figure(figsize=(15,7))
    plt.subplot(221)    
    plt.hist(weights, bins=10)
    plt.title("Distribution of weights")
    
    plt.subplot(222)
    plt.plot(np.cumsum(weights))
    plt.title("Cumulative sum of weights")
    plt.savefig('figures/SBQ weights')
    
    
def plotMMD(samplesSBQ,samplesH,samplesIID,gm,kernel):
    nSBQ = len(samplesSBQ)                         # plot for as many samples as SBQ
    gramH = np.zeros((nSBQ,nSBQ))
    zH = np.zeros((nSBQ))
    fillGram(gramH,zH,kernel,gm,samplesH)          # BQ weights for kernel herding

    gramSBQ = np.zeros((nSBQ,nSBQ))
    zSBQ = np.zeros((nSBQ))
    fillGram(gramSBQ,zSBQ,kernel,gm,samplesSBQ)    # BQ weights for Bayesian quadrature
    
    # MMD of all methods
    mmdsH = []
    mmdsIID = []
    mmdsSBQ = []
    mmdsH_BQ = []
    for i in range(1,nSBQ+1):
        progress(i-1, nSBQ, 'Plot MMD')
        mmdsSBQ.append(MMD2SBQ_Gaussian(kernel,gm,samplesSBQ[:i],gramSBQ[:i,:i],zSBQ[:i]))
        mmdsH_BQ.append(MMD2SBQ_Gaussian(kernel,gm,samplesH[:i],gramH[:i,:i],zH[:i]))
        mmdsH.append(MMD2H_Gaussian(kernel,gm,samplesH[:i]))
        mmdsIID.append(MMD2H_Gaussian(kernel,gm,samplesIID[:i]))

    plt.figure(figsize=(10, 7))
    plt.plot(range(nSBQ),[1/(k+1) for k in range(len(mmdsH))],'--',label="O(1/N)")
    plt.plot(mmdsSBQ,label="SBQ with BQ weights")
    plt.plot(mmdsH_BQ,label="Herding with BQ weights")
    plt.plot(mmdsH,label="Herding with 1/N weights")
    plt.plot(mmdsIID,label="iid sampling")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MMD")
    plt.xlabel("Number of samples")
    plt.ylabel("MMD")
    plt.savefig('figures/MMD')
    plt.show()


def targetRKHS(beta, c, kernel, gm):
    s = 0
    for i in range(len(beta)):
        for j in range(len(gm.weights)):
            s += beta[i] * gm.weights[j] * multivariate_normal.pdf(c[i], gm.means[j], kernel.covariance + gm.covariances[j])
    return s



def generateRKHSFunc(nfuncs, kernel, gm):
    """
        Compute functions in the span of the kernel functions
        and their integrals with respec t to the distribution
    """
    N = 20
    funcs = []
    targets = []
    for _ in range(nfuncs):
        beta = 10 * (np.random.rand(N) + 1)
        c = 4 * 2 * (np.random.rand(N, 2) - 1/2)   # mean for kernel 
        factor = 0
        for k in range(N):
            for l in range(N):
                factor += beta[k] * beta[l] * kernel.pdf(c[k], c[l])
        beta /= (factor)**(1/2)                # in the unit ball the RKHS
        funcs.append(lambda x : np.sum([beta[i] * kernel.pdf(x, c[i]) for i in range(len(beta))], axis=0))
        targets.append(targetRKHS(beta, c, kernel, gm))
    return funcs, targets




def plotErrors(samplesH, samplesSBQ, samplesIID, gm, kernel, funcs, targets):
    """
        Compute functions in the span of the kernel functions
        and their integrals with respec t to the distribution
    """

    nSBQ = len(samplesSBQ)                         # plot for as many samples as SBQ
    gramH = np.zeros((nSBQ,nSBQ))
    zH = np.zeros((nSBQ))
    fillGram(gramH,zH,kernel,gm,samplesH)          # BQ weights for kernel herding

    gramSBQ = np.zeros((nSBQ,nSBQ))
    zSBQ = np.zeros((nSBQ))
    fillGram(gramSBQ,zSBQ,kernel,gm,samplesSBQ)    # BQ weights for Bayesian quadrature

    nfuncs = len(funcs)
    errorsIID = 0; errorsH = 0; errorsH_BQ = 0; errorsSBQ = 0

    for i in range(nfuncs):
        errorsIID += np.abs(np.cumsum(funcs[i](samplesIID)) / np.arange(1, len(samplesH)+1) - targets[i][None])
        errorsH += np.abs(np.cumsum(funcs[i](samplesH)) / np.arange(1, len(samplesH)+1) - targets[i][None])
        errorsH_BQ +=  np.abs(np.cumsum(funcs[i](samplesH) * np.linalg.inv(gramH)@zH) - targets[i][None]) 
        errorsSBQ += np.abs(np.cumsum(funcs[i](samplesSBQ) * np.linalg.inv(gramSBQ)@zSBQ) - targets[i][None]) 

    errorsIID /= nfuncs; errorsH /= nfuncs; errorsH_BQ /= nfuncs; errorsSBQ /= nfuncs
    plt.figure(figsize=(10, 7))
    plt.plot(errorsSBQ,label="SBQ with BQ weights")
    plt.plot(errorsH_BQ,label="Herding with BQ weights")
    plt.plot(errorsH,label="Herding with 1/N weights")
    plt.plot(errorsIID,label="iid sampling")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Mean Absolute Error averaged over %s functions in the RKHS" % nfuncs)
    plt.savefig('figures/Mean Absolute Error')
    plt.show()