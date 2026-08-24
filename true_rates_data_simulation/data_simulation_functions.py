# -*- coding: utf-8 -*-

"""

Functions to simulate data over time.

"""


#### IMPORTS ####

import numpy as np

from particles import distributions as dists

from ctmc_modules.functions import compute_transition_prob_matrix


#### DATA SIMULATION GIVEN TRUE RATES ####


def simulate_data(*, true_rates, n, J, delta_t, y_init):
    """Simulate the data using the true rates."""
    
    K = len(true_rates) - 1
    
    ## k == 0 ##
    
    P_mat = compute_transition_prob_matrix(
        true_rates[0].reshape(-1),
        n,
        delta_t
    )
    
    y_k = np.array([
        dists.Categorical(P_mat[yp_i]).rvs()[0]
        for yp_i in y_init
    ])
    
    data = [y_k.reshape(1, -1)]
    
    ## 1 <= k <= K ##
    
    for k in range(1, K+1):
        
        P_mat = compute_transition_prob_matrix(
            true_rates[k].reshape(-1),
            n,
            delta_t
        )
        
        y_k = np.array([
            dists.Categorical(P_mat[yp_i]).rvs()[0]
            for yp_i in y_k]
        )
        
        data.append(y_k.reshape(1, -1))
    
    return data


def simulate_data_manually_cs1():
    """ Simulate data manually for Case Study 1,
        specifically when J == 1 and K == 300.
    """
    
    data = []
    
    for _ in range(0, 76):
        data.append(np.array([[0]]))
    for _ in range(76, 140):
        data.append(np.array([[1]]))
    for _ in range(140, 176):
        data.append(np.array([[2]]))
    for _ in range(176, 240):
        data.append(np.array([[0]]))
    for _ in range(240, 301):
        data.append(np.array([[1]]))
    
    assert len(data) == 300 + 1 # K + 1
    
    return data

