import numpy as np
import numpy.random as nr
import numpy.linalg as nlin
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import numba
# from numba import njit
# from numba import jitclass, types


# spec1 = [
#     ('means', float64[:]),
#     ('covariances', float64[:,:]),
#     ('weights', float64[:]),
#     ('rvs', types.ListType(types.float64[])),
#     ('similarity_threshhold', float64),
#     ('n_changes', int64)
# ]

# spec2 = [
#     ('spacing', float64),
#     ('n_iterations', int64),
#     ('np_emptyhouses', float64[:, :]),
#     ('np_agenthouses', float64[:, :]),
#     ('similarity_threshhold', float64),
#     ('n_changes', int64)
# ]



# @jitclass(spec1)
class GaussianMixture():
    """
        Class for Gaussian Mixtures object
    """
    def __init__(self, p, mus, sigmas):
        self.means = mus
        self.covariances = sigmas
        self.weights = p
        self.rvs = [multivariate_normal(mus[k], sigmas[k])
                    for k in range(len(self.weights))]

    def pdf(self, X):
        return np.sum([self.weights[i]*self.rvs[i].pdf(X) for i in range(len(self.weights))], axis=0)


class GKernel():
    """
        Gaussian kernel with given covariance
    """
    def __init__(self, mean, sig):
        self.mean = mean
        self.covariance = sig

    def evaluate(self, y):
        return multivariate_normal.pdf(y, mean=self.mean, cov=self.covariance)
    
    def setMean(self, X):
        self.mean = X

    def Gram_matrix(self, X, means):
        """
            X an array of samples
        """
        n = X.shape[0]
        m = means.shape[0]
        gram_matrix = np.zeros((m, m))
        for i in range(n):
            self.mean = means[i, :]
            gram_matrix[i,:] = self.evaluate(X)
        return gram_matrix

# @njit
def herding_sequential_sample(distrib, kernel, n_steps, area, num_samples):
    D = area.shape[0]
    sequence = [np.empty(D)]
    losses = []
    for _ in range(n_steps):
        samples, loss = generate_samples(
            distrib, kernel, sequence, area, num_samples)
        sequence.append(samples)
        losses.append(loss)
    return np.array(sequence[1:]), losses 


def generate_samples(distrib, kernel, sequence, area, num_samples):
    d = area.shape[0]
    samples = area[None, 0] + (area[None, 1] - area[None, 0]) * nr.rand(num_samples, d)
    expected_losses = bmc_variance_next(distrib, kernel, sequence, samples)
    best_idx = np.argmin(expected_losses)
    return samples[best_idx,:], expected_losses[best_idx]

def bmc_variance_next(distrib, kernel, sequence, samples):
    prior_variance = prior_variance_distrib(distrib, kernel)
    existing_samples = np.array(sequence)
    K, D = distrib.means.shape
    num_existing = len(sequence)
    n_samples = samples.shape[0]
    existing_zs = np.zeros(num_existing)
    new_zs = np.zeros(n_samples)
    for k in range(K):
        existing_zs += distrib.weights[k] * multivariate_normal.pdf(existing_samples, distrib.means[k], distrib.covariances[k] + kernel.covariance)
        new_zs += distrib.weights[k] * multivariate_normal.pdf(
            samples, distrib.means[k], distrib.covariances[k] + kernel.covariance)
    # Compute Gram Matrix
    existing_K = kernel.Gram_matrix(existing_samples, existing_samples)
    # All the weights the same for now
    old_weights = np.ones(num_existing) / (num_existing + 1)
    new_weight = 1 / (num_existing + 1)
    old_term = prior_variance - 2 * old_weights.T @ existing_zs + old_weights.T @ existing_K @ old_weights
    # Determine the expected variance if add a new sample
    new_term = np.empty(n_samples)
    for i in range(n_samples):
        kernel.setMean(samples[i,:])
        # print(samples[i,:])
        new_gram_row = kernel.evaluate(existing_samples)
        new_gram_diag = kernel.evaluate(samples[i,:])
        # print((new_gram_col))
        new_term[i] = - 2*new_weight*new_zs[i] + new_weight * 2 * new_gram_col[None] @ old_weights + new_weight**2 * new_gram_diag

    expected_loss = old_term + new_term
    
    return expected_loss



def prior_variance_distrib(distrib, kernel):
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



# def generate_herding(kernel, area, samples):
#     d = area.shape[0]
#     X = area * 

#     x = xb+(xb-xa)*nr.rand()
#     y = yb+(yb-ya)*nr.rand()


