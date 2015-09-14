import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import delta, squared_exponential_cov, covariance_mat
from gp_class_data import x_g, y_g, sample, plot_data, sigmoid
from class_parameters import model_params, common_params
import scipy.optimize as opt


def logistic_loss(f, y):
    """Logistic loss function"""
    f = f.reshape((f.size, 1))
    return (- np.sum(np.log(np.exp(-y * f) + np.ones(y.shape))))


def logistic_likelyhood_hessian(f):
    """Hessian of the p(y|f)"""
    f = f.reshape((f.size, 1))
    diag_vec = (-np.exp(f) / np.square(np.ones(f.shape) + np.exp(f)))
    return np.diag(diag_vec.reshape((diag_vec.size, )))

def logistic_grad(f, y):
    f = f.reshape((f.size, 1))
    return (y / ( np.exp(-y * f)))


#Reassigning the parameters
sigma_f = model_params.sigma_f
sigma_l = model_params.sigma_l
l = model_params.l
density = common_params.density
x0, x1 = common_params.x0, common_params.x1
d, n = common_params.d, common_params.n
m = np.vectorize(lambda x: 0)
K = squared_exponential_cov(sigma_f, sigma_l, l)

#Generating Grid
x1_grid = np.linspace(x0, x1, density) #.reshape((1, density))
x2_grid = np.linspace(x0, x1, density) #.reshape((1, density))
x1_test, x2_test = np.meshgrid(x1_grid, x2_grid)
x1_test = x1_test.reshape((x1_test.size,))
x2_test = x2_test.reshape((x2_test.size,))
x_test = np.array(list(zip(x1_test, x2_test))).T

#Initializing covariance matrices
K_x = covariance_mat(K, x_g, x_g)
K_x_test = covariance_mat(K, x_g, x_test)
K_test_x = covariance_mat(K, x_test, x_g)
K_test = covariance_mat(K, x_test, x_test)

#Optimization
f_0 = np.zeros(y_g.shape)
res = opt.minimize(fun=(lambda f: logistic_loss(f, y_g)), x0=f_0.reshape((f_0.size,)), method='Newton-CG',
                   jac=lambda f: logistic_grad(f, y_g).reshape((f_0.size,)), hess=logistic_likelyhood_hessian, options={'maxiter': 10})
f_opt = res['x']

#Calculating the classification results on the grid
f_test = np.dot(np.dot(K_test_x, np.linalg.inv(K_x)), f_opt)
y_test = sigmoid(f_test.reshape((f_test.size, 1)))
y_test = np.sign(y_test - np.ones(y_test.shape) * 0.5)

#Vizualisation
plot_data(x_g, y_g, 'bo', 'ro')
plt.contour(x1_grid, x2_grid, y_test.reshape((20, 20)), levels = [0.0])
plt.show()
