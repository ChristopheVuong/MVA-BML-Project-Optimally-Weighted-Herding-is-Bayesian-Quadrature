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
    plt.hist(weights,bins=20)
    plt.axvline(0, color='black',linestyle='dashed',linewidth=1,label='zero weight')
    plt.axvline(1/100, color='red',linestyle='dashed',linewidth=1,label='1/n weight')
    plt.xlabel("SBQ weight")
    plt.ylabel("count")
    plt.legend()    
    plt.title("Distribution of weights")
    
    plt.subplot(222)
    plt.title("Cumulative sum of weights")
    plt.plot(np.cumsum(weights),label="sum of BQ weights")
    plt.plot(range(len(weights)),np.ones(len(weights)),'--',label='sum of Herding weights')
    plt.xlabel("number of samples")
    plt.ylabel("sum of weights")
    plt.legend()
    
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


def plotRKHSErrors(samplesH, samplesSBQ, samplesIID, gm, kernel):
    """
        Compute functions in the span of the kernel functions
        and their integrals with respect to the distribution
    """

    N = 20
    nfuncs = 250
    
    nSBQ = len(samplesSBQ)                         # plot for as many samples as SBQ
    gramH = np.zeros((nSBQ,nSBQ))
    zH = np.zeros((nSBQ))
    fillGram(gramH,zH,kernel,gm,samplesH)          # BQ weights for kernel herding

    gramSBQ = np.zeros((nSBQ,nSBQ))
    zSBQ = np.zeros((nSBQ))
    fillGram(gramSBQ,zSBQ,kernel,gm,samplesSBQ)    # BQ weights for Bayesian quadrature
    
    weightsH = zH@np.linalg.inv(gramH)
    weightsSBQ = zSBQ@np.linalg.inv(gramSBQ)
    
    errorsIID = 0; errorsH = 0; errorsH_BQ = 0; errorsSBQ = 0

    for k in range(nfuncs):
        progress(k,nfuncs,'Plot RKHS Errors')
        beta = 10 * (np.random.rand(N) + 1)
        c = 4 * 2 * (np.random.rand(N, 2) - 1/2)   # mean for kernel 
        factor = 0
        for k in range(N):
            for l in range(N):
                factor += beta[k] * beta[l] * kernel.pdf(c[k], c[l])
        beta /= (factor)**(1/2)                # in the unit ball the RKHS
        f = lambda x : np.sum([beta[i] * kernel.pdf(x, c[i]) for i in range(len(beta))], axis=0)
        target = targetRKHS(beta, c, kernel, gm)
        errorsIID += np.abs(np.cumsum(f(samplesIID)) / np.arange(1, len(samplesIID)+1) - target[None])
        errorsH += np.abs(np.cumsum(f(samplesH)) / np.arange(1, len(samplesH)+1) - target[None])
     
        errorsH_BQ +=  np.abs(np.cumsum(f(samplesH) * weightsH) - target[None]) 
        errorsSBQ += np.abs(np.cumsum(f(samplesSBQ) * weightsSBQ) - target[None]) 

    errorsIID /= nfuncs; errorsH /= nfuncs; errorsH_BQ /= nfuncs; errorsSBQ /= nfuncs

    plt.figure(figsize=(10, 7))
    plt.plot(errorsSBQ,label="SBQ with BQ weights")
    # plt.plot(errorsH_BQ,label="Herding with BQ weights")
    plt.plot(errorsH,label="Herding with 1/N weights")
    plt.plot(errorsIID,label="iid sampling")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Mean Absolute Error averaged over %s functions in the RKHS" % nfuncs)
    plt.savefig('figures/Mean Absolute Error')
    plt.show()


def targetOutRKHS(beta, c, gm, covs):
    s = 0
    for i in range(len(beta)):
        for j in range(len(gm.weights)):
            s += beta[i] * gm.weights[j] * multivariate_normal.pdf(c[i], gm.means[j], covs[i] + gm.covariances[j])
    return s

def plotOutErrors(samplesH, samplesSBQ, samplesIID, gm, kernel):
    """
        Compute functions outside the RKHS (Gaussian densities with random covariances) 
        and their integrals with respect to the distribution
    """

    N = 20
    nfuncs = 250
    D = 2 # dim
    
    nSBQ = len(samplesSBQ)                         # plot for as many samples as SBQ
    gramH = np.zeros((nSBQ,nSBQ))
    zH = np.zeros((nSBQ))
    fillGram(gramH,zH,kernel,gm,samplesH)          # BQ weights for kernel herding

    gramSBQ = np.zeros((nSBQ,nSBQ))
    zSBQ = np.zeros((nSBQ))
    fillGram(gramSBQ,zSBQ,kernel,gm,samplesSBQ)    # BQ weights for Bayesian quadrature
    
    weightsH = zH@np.linalg.inv(gramH)
    weightsSBQ = zSBQ@np.linalg.inv(gramSBQ)
    
    errorsIID = 0; errorsH = 0; errorsH_BQ = 0; errorsSBQ = 0

    stdsig = 2      # parameter for covariance matrix
    ratio = [.1,2]  # parameter for covariance matrix

    for k in range(nfuncs):
        progress(k,nfuncs,'Plot Out Errors')
        beta = 2 * (np.random.rand(N) + 1)
        c = 4 * 2 * (np.random.rand(N, 2) - 1/2)   # mean for kernel
        covs = np.zeros((N, D, D))
        
        for k in range(N):
            sig = stdsig * (np.random.rand(D)*(ratio[1] - ratio[0]) + ratio[0])
            covs[k] = np.diag(sig**(-2))
            U, _ = np.linalg.qr(np.random.randn(D,D))
            covs[k] = U @ covs[k] @ U.T

        f = lambda x : np.sum([beta[i] * multivariate_normal.pdf(x, c[i], covs[i]) for i in range(len(beta))], axis=0)

        target = targetOutRKHS(beta, c, gm, covs)
        errorsIID += np.abs(np.cumsum(f(samplesIID)) / np.arange(1, len(samplesIID)+1) - target[None])
        errorsH += np.abs(np.cumsum(f(samplesH)) / np.arange(1, len(samplesH)+1) - target[None])
        errorsH_BQ +=  np.abs(np.cumsum(f(samplesH) * weightsH) - target[None]) 
        errorsSBQ += np.abs(np.cumsum(f(samplesSBQ) * weightsSBQ) - target[None]) 

    errorsIID /= nfuncs; errorsH /= nfuncs; errorsH_BQ /= nfuncs; errorsSBQ /= nfuncs

    plt.figure(figsize=(10, 7))
    plt.plot(errorsSBQ,label="SBQ with BQ weights")
    # plt.plot(errorsH_BQ,label="Herding with BQ weights")
    plt.plot(errorsH,label="Herding with 1/N weights")
    plt.plot(errorsIID,label="iid sampling")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Mean Absolute Error averaged over %s functions outside the RKHS" % nfuncs)
    plt.savefig('figures/Mean Out Error')
    plt.show()
