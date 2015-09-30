from covariance_functions import squared_exponential_cov, matern_cov, gamma_exponential_cov, rational_quadratic_cov,\
    add_noise
import numpy as np
# class DataParameters:
#     """class, containing the parameters of data being generated"""
#     def __init__(self, sigma_f, sigma_l, l):
#         self.sigma_f, self.sigma_l, self.l = sigma_f, sigma_l, l


class CommonParameters:
    """class, containg the common parameters of the data"""
    def __init__(self, num_of_examples, dim, x0, x1, density):
        self.n, self.d, self.x0, self.x1, self.density = num_of_examples, dim, x0, x1, density


class ModelParameters:
    """class, containing the hyper-parameters of the model prior distribution"""
    def __init__(self, family, noise_var, data_seed=None):
        if data_seed == None:
            data_seed = np.random.rand()
        self.cov_func, self.ml_grad, self.noise_var, self.data_seed = \
            add_noise(family.covariance_function, noise_variance), family.marginal_likelyhood_grad, \
            noise_var, data_seed



noise_variance = 0.0
K1 = squared_exponential_cov(sigma_f=1.0, l=0.4)
K2 = matern_cov(nu=1.5, l=0.9)
K3 = gamma_exponential_cov(gamma=0.5, l=0.9)
K4 = rational_quadratic_cov(alpha=1.5, l=0.9)
common_params = CommonParameters(num_of_examples=100, dim=2, x0=-5, x1=5, density=20)
# data_params = ModelParameters(squared_exponential_cov(sigma_f=1.0, l=0.4), noise_var=noise_variance)
data_params = ModelParameters(K1, noise_var=noise_variance, data_seed=5)
model_params = ModelParameters(K4, noise_var=noise_variance)
