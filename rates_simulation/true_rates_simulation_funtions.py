"""
Functions to simulate true rates over time.

"""


import numpy as np
from ctmc_modules.ctmc_ssms import compute_transition_prob_matrix
from particles import distributions as dists


def sigmoid(x, a, b, m, k):
    """ result = a when t = 0. """
    result = a + (b-a) / (1 + np.exp(-k*(x-m)))
    result -= a + (b-a) / (1 + np.exp(k * m))
    result += a
    return result


def simulate_sigmoid_growth(mu0, max_growth, n, J, K, delta_t, sig_func_val=14):
    """ Simulate true_states and data where rates undergo Sigmoid growth. """
    
    L = mu0.shape[0]

    # k = 0
    true_states = [mu0.reshape(1, -1)]
    data = [np.sort(np.array([j % n for j in range(J)])).reshape(1, -1)]
    
    # k = 1, ..., K
    for k in range(1, K+1):
        # lams
        lams_k = np.array([sigmoid(k, mu0[i], mu0[i] + max_growth[i],
                                   K/2, sig_func_val / K)
                           for i in range(L)])
        true_states.append(lams_k.reshape(1, -1))
        
        # y
        P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n, delta_t)
                          for cur_lams in true_states[-2]], axis=0)
        y_t = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
                        for y_i in data[-1]])
        data.append(y_t)
    return true_states, data


def simulate_constant_rates(mu0, n, J, K, delta_t):
    """ Simulate true_states and data where rates remain constant. """
    
    # k = 0, ..., K
    true_states = [ mu0.reshape(1, -1) for _ in range(K+1) ]
    
    # k = 0
    data = [ np.sort(np.array([j % n for j in range(J)])).reshape(1, -1) ]

    # k = 1, ..., K
    for k in range(1, K+1):
        # y
        P_mat = np.stack([compute_transition_prob_matrix(cur_lams, n, delta_t)
                          for cur_lams in true_states[k-1]], axis=0)
        y_k = np.array([dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i]).rvs()
                        for y_i in data[-1]])
        # y_k = np.array([dists.Categorical(P_mat[:, y_i]).rvs()
        #                 for y_i in data[-1]])
        data.append(y_k)
    return true_states, data

