import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import delta, squared_exponential_cov, covariance_mat
from gp_reg_data import x_g, y_g, sample, plot_data
from reg_parameters import model_params, common_params


def sample_for_matrices(mean_vec, cov_mat):
    y = np.random.multivariate_normal(mean_vec.reshape((mean_vec.size,)), cov_mat)
    upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
    lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
    return (y, upper_bound, lower_bound)

#Reassigning the parameters
sigma_f = model_params.sigma_f
sigma_l = model_params.sigma_l
l = model_params.l
density = common_params.density
x0, x1 = common_params.x0, common_params.x1
d, n = common_params.d, common_params.n
m = np.vectorize(lambda x: 0)
K = squared_exponential_cov(sigma_f, sigma_l, l)

#Fitting
x_test = np.linspace(x0, x1, density).reshape((1, density))
K_x = covariance_mat(K, x_g, x_g)
K_x_test = covariance_mat(K, x_g, x_test)
K_test_x = covariance_mat(K, x_test, x_g)
K_test = covariance_mat(K, x_test, x_test)
I = np.eye(K_x.shape[0])
anc_mat = np.linalg.inv(K_x + I * sigma_l**2)
new_mean = np.dot(np.dot(K_test_x, anc_mat), y_g)
new_cov = K_test - np.dot(np.dot(K_test_x, anc_mat), K_x_test)
y_test, u_b, l_b = sample_for_matrices(new_mean, new_cov)

#Plotting
plot_data(x_g, y_g, 'x')
plot_data(x_test, u_b, 'r')
plot_data(x_test, l_b, 'g')
plot_data(x_test, y_test, 'b')
plt.show()