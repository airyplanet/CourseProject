import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import delta, squared_exponential_cov, covariance_mat
from reg_parameters import data_params, common_params


def plot_data(x, y, color):
    x1 = x.reshape((x.size, ))
    y1 = y.reshape((y.size, ))
    plt.plot(x1, y1, color)


def sample(mean_func, cov_func, x):
    """returns a sample of a gaussian process for given mean and covariance at given points"""
    cov_mat = covariance_mat(cov_func, x, x)
    m_v = mean_func(x)
    mean_vector = m_v.reshape((m_v.size,))
    y = np.random.multivariate_normal(mean_vector, cov_mat)
    return y


#Reassigning the parameters
sigma_f = data_params.sigma_f
sigma_l = data_params.sigma_l
l = data_params.l
density = common_params.density
x0, x1 = common_params.x0, common_params.x1
d, n = common_params.d, common_params.n
m = np.vectorize(lambda x: 0)
K = squared_exponential_cov(sigma_f, sigma_l, l)

#Producing data
x_g = x0 + np.random.rand(d, n)*(x1 - x0)
y_g = sample(m, K, x_g)
y_g = y_g.reshape((y_g.size, 1))

if __name__ == "__main__":
    plot_data(x_g, y_g, 'x')
    plt.show()
