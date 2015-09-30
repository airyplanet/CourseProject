from covariance_functions import SquaredExponential, GammaExponential
import numpy as np
# class DataParameters:
#     """class, containing the parameters of data being generated"""
#     def __init__(self, covariance_function):
#         self.cov_func = covariance_function


class CommonParameters:
    """class, containg the common parameters of the data"""
    def __init__(self, num_of_examples, dim, density):
        self.n, self.d, self.density = num_of_examples, dim, density


class ModelParameters:
    """class, containing the hyper-parameters of the model prior distribution"""
    def __init__(self, family, noise_var, data_seed=None):
        if data_seed == None:
            data_seed = np.random.rand()
        self.cov_obj, self.noise_var, self.data_seed = \
            family, noise_var, data_seed

noise_variance = 0.2
oracle1 = SquaredExponential(sigma_f=1.0, l=0.4, noise=noise_variance)
oracle2 = SquaredExponential(sigma_f=2.0, l=0.7, noise=noise_variance)
# oracle3 = GammaExponential(sigma_f=2.5, l=0.3, gamma=2.1, noise=0.3)
common_params = CommonParameters(num_of_examples=74, dim=27, density=100)
data_params = ModelParameters(oracle1, noise_var=noise_variance, data_seed=5)
model_params = ModelParameters(oracle2, noise_var=noise_variance)
