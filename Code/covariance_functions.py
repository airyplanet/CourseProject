import numpy as np


def delta(x, y):
    """Delta-function"""
    return max(x == y)


def squared_exponential_cov(alpha, sigma, l):
    """Squared exponential (SE) covariance function"""
    def se_cov(x, y):
        return np.exp(-np.square(np.linalg.norm(x - y) / (2*l**2))) * alpha + sigma**2 * delta(x, y)
    return se_cov


def covariance_mat(covariance_func, x, y):
    """Computing covariance matrix for given covariance function and point arrays"""
    mat = np.zeros((x.shape[1], y.shape[1]))
    for i in range(0, x.shape[1]):
        for j in range(0, y.shape[1]):
            mat[i, j] = covariance_func(x[:, i], y[:, j])
    return mat
