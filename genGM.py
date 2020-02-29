import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
import scipy as scp
from scipy.stats import multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt

class GaussianMixture():
    def __init__(self, p, mus, sigmas):
        self.means = mus
        self.covariances = sigmas
        self.weights = p
        self.rvs = [multivariate_normal(mus[k], sigmas[k]) for k in range(len(self.weights))]

    def pdf(self, X):
        return np.sum([self.weights[i]*self.rvs[i].pdf(X) for i in range(len(self.weights))], axis=0)


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
    mus = (2*nr.rand(K, D) - 1) * stdmu
    sigmas = np.zeros((K, D, D))
    for k in range(K):
        sig = stdsig * (nr.rand(D)*(ratio[1] - ratio[0]) + ratio[0])
        sigmas[k] = np.diag(sig**(-2))
        U, R = nlin.qr(nr.randn(D,D))
        sigmas[k] = U @ sigmas[k] @ U.T
    p = nr.rand(K)
    p /= np.sum(p)
    GM_obj = GaussianMixture(p, mus, sigmas)
    
    xx, yy = np.mgrid[np.min(
    mus[:,0])-stdsig:np.max(mus[:,0])+stdsig:.1, np.min(mus[:,1])-stdsig:np.max(mus[:,1])+stdsig:.1]
    # xx, yy = np.mgrid[-4:6:.1, -4:6:.1]
    pos = np.empty(xx.shape + (2,))
    pos[:,:,0] = xx; pos[:,:,1] = yy
    
    f = GM_obj.pdf(pos)
    
    if D == 2 and Plot:

        # print(f.shape)
        plt.contour(xx, yy, f, 20)
        plt.scatter(*mus.T, facecolors='none', edgecolors='r')
        plt.show()

    return GM_obj


# def GM_obj(mus, sigmas):
#     rvs = []
#     for k in range(mus.shape[0]):
#         rvs.append(multivariate_normal(mus[k], sigmas[k]))
#     return rvs

# generate_GM(20, 2)
