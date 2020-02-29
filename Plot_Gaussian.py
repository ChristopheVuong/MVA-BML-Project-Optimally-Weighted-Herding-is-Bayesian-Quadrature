
import matplotlib.pyplot as plt
import numpy as np


def plotContour(gm,samples=None):
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
    if samples is not None:
        plt.scatter(*samples.T,c='red',marker='x')
    plt.show()
