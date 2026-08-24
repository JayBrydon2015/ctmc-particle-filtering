# -*- coding: utf-8 -*-

"""

Functions to simulate true rates over time.

"""


#### IMPORTS ####

import numpy as np
from scipy.special import expit

from ctmc_modules.functions import gen_pos_to_lams_idx


#### CONSTANT RATES ####


def simulate_constant_rates(*, mu0, K):
    """ Simulate true_states and data where rates remain constant. """
    return [ mu0.reshape(1, -1) for _ in range(K+1) ]


#### SIGMOID EVOLUTION ####


def sigmoid(x, a, b, m, s):
    """ Sigmoid function. Note: result = a when x = 0. """
    # expit(u) == 1 / (1 + np.exp(-u))
    result = a + (b - a) * expit(s * (x - m))
    result -= a + (b - a) * expit(-s * m)
    result += a
    return result


def simulate_sigmoid_growth(*, K, mu0=[1, 1], max_growth=[7, 7], s=1):
    """ Simulate the true transition rates which undergo sigmoid growth. """
    L = len(mu0)
    return [
        np.array([
            sigmoid(k, mu0[i], mu0[i] + max_growth[i], K/2, s)
            for i in range(L)
        ]).reshape(1, -1)
        for k in range(K+1)
    ]


#### CASE STUDY #1: 3 NON-ZERO CONSTANT RATES & 3 STATES ####


def simulate_rates_cs1(*, K, epsilon=1, delta=1, phi=4):
    """ Simulate rates according to Case Study 1. """
    lams_k = np.array([epsilon, 0, 0, delta, phi, 0]).reshape(1, -1)
    return [ lams_k for _ in range(K+1) ]


#### SINUSOIDAL RATES (USED IN CASE STUDY #2) ####


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
        phi = np.array([np.pi / 4, np.pi / 2])
    if a is None:
        a = np.array([0.5, 1])
    if b is None:
        b = np.array([3.5, 4])
    if s is None:
        s = np.array([1, 1])
    
    return [
        np.maximum(
            (b - a) * np.sin(np.pi * s * k / K - phi) ** 2 + a
            , 0).reshape(1, -1)
        for k in range(K+1)
    ]


#### CASE STUDY #3: ABRUPTLY CHANGING RATES ####


def sudden_jump_function(x, a, b, m):
    """Piecewise function."""
    return a if x < m else b


def simulate_rates_cs3(*, K, a=[1, 1], b=[20, 20]):
    """ Simulate the two true transition rates which undergo an abrupt
        change (Case Study #3).
    """
    L = len(a)
    return [
        np.array([
            sudden_jump_function(k, a[i], b[i], K/2)
            for i in range(L)
        ]).reshape(1, -1)
        for k in range(K+1)
    ]


#### CASE STUDY #4: HIGH-DIMENSIONAL STATE SPACE ####


def sigmoid_cs4(x):
    return expit(4 * x)


def simulate_rates_cs4(K: int):
    """ Simulate true rates for Case Study #4 on a high-dimensional CTMC. """
    
    n = 5
    num_lams = n * (n-1)
    
    non_zero_transition_rates_positions = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)
    ]
    
    true_rates = []
    
    for k in range(K + 1):
        
        true_rates_k = np.zeros(num_lams)
        
        # L_1_2
        p, q = non_zero_transition_rates_positions[0]
        lam_idx = gen_pos_to_lams_idx(p, q, n)
        L_1_2 = 1 + 2 * sigmoid_cs4(- k / K)
        true_rates_k[lam_idx] = L_1_2
        
        # L_2_3
        p, q = non_zero_transition_rates_positions[1]
        lam_idx = gen_pos_to_lams_idx(p, q, n)
        L_2_3 = 0.5 * np.sin(2 * np.pi * k / K) + 2
        true_rates_k[lam_idx] = L_2_3
        
        # L_3_4
        p, q = non_zero_transition_rates_positions[2]
        lam_idx = gen_pos_to_lams_idx(p, q, n)
        L_3_4 = 4 * sigmoid_cs4(k / K)
        true_rates_k[lam_idx] = L_3_4
        
        # L_4_5
        p, q = non_zero_transition_rates_positions[3]
        lam_idx = gen_pos_to_lams_idx(p, q, n)
        L_4_5 = 1 + 2 * sigmoid_cs4(k / K)
        true_rates_k[lam_idx] = L_4_5
        
        # L_5_1
        p, q = non_zero_transition_rates_positions[4]
        lam_idx = gen_pos_to_lams_idx(p, q, n)
        L_5_1 = 2
        true_rates_k[lam_idx] = L_5_1
        
        true_rates.append(true_rates_k.reshape(1, -1))
    
    return true_rates

