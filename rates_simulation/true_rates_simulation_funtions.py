"""
Functions to simulate true rates over time.

"""


import numpy as np
from ctmc_modules.ctmc_ssms import compute_transition_prob_matrix
from particles import distributions as dists


def sigmoid(x, a, b, m, s):
    """ result = a when t = 0. """
    result = a + (b-a) / (1 + np.exp(-s*(x-m)))
    result -= a + (b-a) / (1 + np.exp(s * m))
    result += a
    return result


def simulate_sigmoid_growth(*, mu0, max_growth, K, sig_func_val=14):
    """ Simulate true_states and data where rates undergo Sigmoid growth. """
    L = mu0.shape[0]
    return [
        np.array([sigmoid(k, mu0[i], mu0[i] + max_growth[i],
                          K/2, sig_func_val / K)
                  for i in range(L)]).reshape(1, -1)
        for k in range(K+1)
    ]


def simulate_constant_rates(*, mu0, K):
    """ Simulate true_states and data where rates remain constant. """
    return [ mu0.reshape(1, -1) for _ in range(K+1) ]


def simulate_data(*, true_rates, n, J, delta_t, y_init):
    """ Simulate the data using the true rates. """
    K = len(true_rates) - 1
    
    # k == 0
    P_mat = compute_transition_prob_matrix(true_rates[0].reshape(-1),
                                           n, delta_t)
    y_k = np.array([dists.Categorical(P_mat[yp_i]).rvs()[0]
                    for yp_i in y_init])
    data = [y_k.reshape(1, -1)]
    
    # 1 <= k <= K
    for k in range(1, K+1):
        P_mat = compute_transition_prob_matrix(true_rates[k].reshape(-1),
                                               n, delta_t)
        y_k = np.array([dists.Categorical(P_mat[yp_i]).rvs()[0]
                        for yp_i in y_k])
        data.append(y_k.reshape(1, -1))
    return data

