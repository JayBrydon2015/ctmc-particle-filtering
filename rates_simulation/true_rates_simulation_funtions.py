"""
Functions to simulate true rates over time.

"""


import numpy as np
from ctmc_modules.ctmc_ssms import compute_transition_prob_matrix
from particles import distributions as dists


## SIGMOID GROWTH RATES ##

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


## CONSTANT RATES ##

def simulate_constant_rates(*, mu0, K):
    """ Simulate true_states and data where rates remain constant. """
    return [ mu0.reshape(1, -1) for _ in range(K+1) ]


## SINE RATES ##

def simulate_sine_rates_n2(*, K, phi=None, a=None, b=None, s=None):
    """ Simulate true_rates that follow a sine squared wave.
            NOTE: for 2 states (n == 2).
        
        lams_k = (b - a) * sin^2(pi * s * k / K - phi) + a
        
        - Each value in phi (the phase shift) only needs to be between
          0 and pi (np.pi).
        - a is the array of minimum values for each lam; b the maximum.
        - s controls the period of the waves (how wide or narrow they are). A
          value of 1 means that the wave completes one cycle between t_0 and
          t_K.
    """
    
    if phi is None:
        phi = np.array([0, np.pi / 2])
    if a is None:
        a = np.array([0, 1])
    if b is None:
        b = np.array([3, 4])
    if s is None:
        s = np.array([1, 1])
    
    return [
        np.maximum(
            (b - a) * np.sin(np.pi * s * k / K - phi) ** 2 + a
            , 0).reshape(1, -1)
        for k in range(K+1)
    ]


## EXAMPLE A: 3 CONT RATES & 3 STATES ##

def simulate_example_a(*, K, epsilon=1, delta=1, phi=4):
    """ Simulate rates according to Example A (in thesis). """
    lams_k = np.array([epsilon, 0, 0, delta, phi, 0]).reshape(1, -1)
    return [ lams_k for _ in range(K+1) ]


## DATA SIMULATION GIVEN TRUE RATES ##

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

