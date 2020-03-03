#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from Herding import *
from SBQ import *


def plotContour(gm,samplesH=None,samplesSBQ=None):
    stdsig = 2

    xx, yy = np.mgrid[np.min(
        gm.means[:, 0])-stdsig:np.max(gm.means[:, 0])+stdsig:.1, np.min(gm.means[:, 1])-stdsig:np.max(gm.means[:, 1])+stdsig:.1]
    # xx, yy = np.mgrid[-4:6:.1, -4:6:.1]
    pos = np.empty(xx.shape + (2,))
    pos[:,:,0] = xx; pos[:,:,1] = yy

    f = gm.pdf(pos)
    plt.figure()
    plt.contour(xx, yy, f, 20)
    # plt.scatter(*mus.T, facecolors='none', edgecolors='r')
    if samplesSBQ is not None:
        plt.scatter(*samplesSBQ.T,c='black',marker='o',label='SBQ Samples')
    if samplesH is not None:
        plt.scatter(*samplesH.T,c='red',marker='x',label='Herding Samples')
    if samplesH is not None or samplesSBQ is not None:
        plt.legend()
    plt.show()

    
    
def plotMMD(samplesSBQ,samplesH,samplesIID,gm,kernel):
    gramH = np.zeros((100,100))
    zH = np.zeros((100))
    fillGram(gramH,zH,kernel,gm,samplesH)

    gramSBQ = np.zeros((100,100))
    zSBQ = np.zeros((100))
    fillGram(gramSBQ,zSBQ,kernel,gm,samplesSBQ)
    
    mmdsH = []
    mmdsIID = []
    mmdsSBQ = []
    mmdsH_BQ = []
    for i in range(1,len(samplesSBQ)):
        print(i)
        mmdsSBQ.append(MMD2SBQ_Gaussian(kernel,gm,samplesSBQ[:i],gramSBQ[:i,:i],zSBQ[:i]))
        mmdsH_BQ.append(MMD2SBQ_Gaussian(kernel,gm,samplesH[:i],gramH[:i,:i],zH[:i]))
        mmdsH.append(MMD2H_Gaussian(kernel,gm,samplesH[:i]))
        mmdsIID.append(MMD2H_Gaussian(kernel,gm,samplesIID[:i]))


    plt.plot(range(len(mmdsH)),[1/(k+1) for k in range(len(mmdsH))],'--',label="O(1/N)")
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
    plt.show()

