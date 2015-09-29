from sklearn.datasets import load_svmlight_file
import numpy as np
from numpy import random
from sklearn import svm

#Parameters
random_seed_w0 = 32
mu, sigma1 = 0, 10

x_g, y_g = load_svmlight_file('Data/pyrim.txt')
x_g = x_g.T
x_g = x_g.toarray()
# y_g = y_g.toarray()
data_name = 'Pyrim'

y_g = y_g.reshape((y_g.size, 1))
x_test = x_g[:, int(x_g.shape[0] * 0.8):]
y_test = y_g[int(x_g.shape[0] * 0.8):, :]
y_g = y_g[:int(x_g.shape[0] * 0.8), :]
x_g = x_g[:, : int(x_g.shape[0] * 0.8)]

print(x_g.shape, y_g.shape)
#Generating the starting point
np.random.seed(random_seed_w0)
w0 = np.random.rand(3)

clf = svm.SVC()
clf.fit(x_g.T, y_g.reshape((y_g.size,)))
svm_y_test = clf.predict(x_test.T)
print("SVM: ", np.linalg.norm(y_test - svm_y_test))