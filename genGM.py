import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_GM(K, D, Plot=True):
    """
        Generate Mixture of Gaussians
        Plot the levels and the means of the Gaussians
    """
    K = 20          # mixture components
    stdmu = 4       # parameter for mean
    stdsig = 2      # parameter for covariance matrix
    ratio = [.1,2]  # parameter for covariance matrix
    # al = K          # parameter for pi
    mus = (2*np.random.rand(K, D) - 1) * stdmu
    sigmas = np.zeros((K, D, D))
    for k in range(K):
        sig = stdsig * (np.random.rand(D)*(ratio[1] - ratio[0]) + ratio[0])
        sigmas[k] = np.diag(sig**(-2))
        U, _ = np.linalg.qr(np.random.randn(D,D))
        sigmas[k] = U @ sigmas[k] @ U.T
    p = np.random.rand(K)
    p /= np.sum(p)
    GM_obj = GaussianMixture(p, mus, sigmas)
    
    if D == 2 and Plot:
        xx, yy = np.mgrid[np.min(
        mus[:,0])-stdsig:np.max(mus[:,0])+stdsig:.1, np.min(mus[:,1])-stdsig:np.max(mus[:,1])+stdsig:.1]
        # xx, yy = np.mgrid[-4:6:.1, -4:6:.1]
        pos = np.empty(xx.shape + (2,))
        pos[:,:,0] = xx; pos[:,:,1] = yy
        f = GM_obj.pdf(pos)

        plt.contour(xx, yy, f, 20)
        plt.scatter(*mus.T, facecolors='none', edgecolors='r')
        plt.show()

    return GM_obj