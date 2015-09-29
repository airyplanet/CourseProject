import time
import numpy as np
import matplotlib.pyplot as plt


class Problem:
    """ This is a class, containing the information about the problem """
    def __init__(self, oracle_function, starting_point, solution_loss=np.NaN):
        """
        :param oracle_function: (loss, grad)
        :param starting_point:
        :param solution_loss: loss at the solution if given
        :return: problem object
        """
        self.w0, self.true_loss = starting_point, solution_loss
        self.oracle = oracle_function

def full_gradient_descent(prb, max_iter=1000, max_time=np.inf, freq=5):
    """
    :param prb: Problem class object
    :param max_iter: maximum number of iterations
    :param max_time: maximum time
    :param freq: printing frequence
    :return: (vector of points at each iteration, vector of times at each iteration)
    """
    #Retrieving the parameters
    w0 = prb.w0
    oracle = prb.oracle

    #Resetting the counters
    w = w0
    iteration_counter = 0
    point_vec = []
    start = time.clock()
    time_vec = []
    loss_vec = []

    #Armiho rule for linear search
    a_eps, a_theta = 0.0001, 0.1

    def armiho(w, stp):
        step = stp
        # step /= a_theta
        current_loss, gradient = oracle(w)
        w_new = w - step * gradient
        while (w_new <= 0).any():
            step *= a_theta
            w_new = w - step * gradient
        new_loss, new_gradient = oracle(w_new)
        while (new_loss > current_loss - a_eps * step * np.linalg.norm(gradient)**2) or np.isnan(new_loss):
            step *= a_theta
            # print("Step, w_new: ", step, w_new, new_loss, current_loss)
            # print("Gradient: ", gradient)
            w_new = w - step * gradient
            while (w_new <= 0).any():
                step *= a_theta
                w_new = w - step * gradient
            new_loss, new_gradient = oracle(w_new)
        return step

    step_rule = armiho
    step = 1
    loss, gradient = oracle(w)
    while (iteration_counter < max_iter) and (time.clock() - start < max_time):
        if iteration_counter % freq == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            loss_vec.append(loss)
            print("FG Iteration ", iteration_counter)
            print(step, np.linalg.norm(gradient), w)
            # print(w)
        step = 10e-4#step_rule(w, step)
        w_new = w - step * gradient
        # while (w_new <= 0).any():
        #     step *= a_theta
        #     w_new = w - step * gradient
        #     print(w_new, step)
        w = w_new
        iteration_counter += 1
        loss, gradient = oracle(w)
        # print(np.linalg.norm(gradient), step)
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec, loss_vec)
