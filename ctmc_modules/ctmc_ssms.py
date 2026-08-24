# -*- coding: utf-8 -*-

"""

CTMC state-space models (SSMs) as Python objects (classes).

"""


#### IMPORTS ####

import numpy as np

from particles import augmented_state_space_models as augssm
from particles import distributions as dists

from ctmc_modules.ctmc_config import GAMMA_SSM_TV

from ctmc_modules.functions import (
    get_gamma_params_from_mean_var,
    gen_to_lams,
    get_default_rw_initial_config,
    compute_transition_prob_matrix,
    get_cat_dist
)



## CTMC SSM CLASSES ##



class CTMC(augssm.AugmentedStateSpaceModel):
    r"""
        CTMC Augmented SSM

        ----- Parameters -----
        n: number of states.
        J: number of random walkers.
        delta_t: real time between observations.
        C: to scale the transition variance by.
        a0: Gamma dist alpha parameters for PX0 in (n, n) ndarray.
        b0: Gamma dist beta parameters for PX0 in (n, n) ndarray.
        y_init: initial configuration of RWs (set to None or don't pass 
          into initialisation if using the default).
        tv_type: decides which transition variance to use (see PX). If
          "PV", proportional variance (PV); if "CV", constant variance (CV).
        px_reg_term: a small constant added to beta in Option 2 in PX to help
          with numberical stability. Stops beta from being too close to 0.
          Note that the mean of a Gamma distribution is alpha / beta.
        px_verbose: if True and if t % 20 == 0, prints the value of t in PX.

        ----- Notes -----
        - Track the list of rates/lams rather than generator A.
        - lams: ndarray of shape (n*(n-1), ). All rates in a 1D array, ordered
          as 12, ..., 1n, 21, 23, ..., 2n, ..., n(n-2), n(n-1).
        - y defined as in type 4.
        - y_init: ndarray of shape (J, ).
        - SSM starts with the RWs spread across the states as evenly as
          possible by default (y_init == None).
        - y_k (data[k]): ndarray of shape (1, J), by convention.
        - CTMC states are 0 to n-1 in the code itself, however. This is
          because dists.Categorical returns values between 0 and n-1.
    """
    
    def __init__(self, *,
            n: int,
            J: int,
            delta_t: float,
            tv_type: GAMMA_SSM_TV,
            C: float,
            mu0,
            var0,
            y_init = None,
            reg_term: float = 1e-6,
            px_verbose: bool = False,
            px_verbose_int: int = 20
        ):
        
        assert n >= 2 and J >= 1 and delta_t > 0
        self.n = n
        self.J = J
        self.delta_t = delta_t
        
        self.tv_type = tv_type
        
        if C <= 0:
            raise ValueError(
                f"C must be positive. Currently, its value is {C}."
            )
        self.C = C
        
        self.mu0  = mu0
        self.var0 = var0
        if len(self.mu0.shape) > 1: # Assuming currently a (n, n) ndarray
            self.mu0 = gen_to_lams(self.mu0)
        if len(self.var0.shape) > 1:  # Assuming currently a (n, n) ndarray
            self.var0 = gen_to_lams(self.var0)
        
        self.a0, self.b0 = get_gamma_params_from_mean_var(mu0, var0)
        assert (
            self.a0.shape[0] == self.n * (self.n - 1) and
            self.a0.shape    == self.b0.shape
        )
        self.num_lams = self.n * (self.n - 1)
        
        if y_init is None:
            self.y_init = get_default_rw_initial_config(self.n, self.J)
        else:
            self.y_init = y_init
        
        self.reg_term = reg_term
        
        self.px_verbose = px_verbose
        self.px_verbose_int = px_verbose_int
    
    
    def PX0(self):
        return dists.IndepProd(
            *[dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        )
    
    
    def PX(self, t, xp):
        
        if self.px_verbose and t % self.px_verbose_int == 0:
            print("t:", t)
        
        delta_t_times_C = self.delta_t * self.C
        
        if self.tv_type == "PV":
            ## PV (default): Var = lambda * DELTA_T * C ##
            alpha  = xp / delta_t_times_C
            beta_i = np.repeat(1 / delta_t_times_C, alpha.shape[0])
            lams_dists = [
                dists.Gamma(alpha[:, l], beta_i)
                for l in range(alpha.shape[1])
            ]
            
        elif self.tv_type == "CV":
            ## CV: Var = DELTA_T * C ##
            alpha = xp ** 2 / delta_t_times_C
            beta  = xp / delta_t_times_C + self.reg_term # Forces beta != 0
            lams_dists = [
                dists.Gamma(alpha[:, l], beta[:, l])
                for l in range(alpha.shape[1])
            ]
            
        else:
            raise NotImplementedError
        
        return dists.IndepProd(*lams_dists)
    
    
    def PY(self, t, xp, x, datap=None):
        
        if datap is None: # t == 0
            datap = self.y_init
        else: # t >= 1, datap == data[t-1]
            # By convention, datap is originally of shape (1, J)
            # Reshape it to (J, )
            datap = datap.reshape(-1)
        
        P_mat = np.stack(
            [
                compute_transition_prob_matrix(
                    cur_lams,
                    self.n,
                    self.delta_t
                )
                for cur_lams in x
            ],
            axis=0
        )
        
        return dists.IndepProd(
            *[get_cat_dist(P_mat, y_i) for y_i in datap]
        )



class GP_CTMC(augssm.AugmentedStateSpaceModel):
    r"""
        CTMC Augmented SSM with the log-rates modelled via
        a Gaussian process with an OU covariance function/kernel
        multiplied by the marginal variance sigma^2.

        ----- Parameters -----
        n: number of states.
        J: number of random walkers.
        delta_t: real time between observations.
        kappa: the output of the OU kernel when |t - s| = delta_t.
            It is in (0, 1] and is proportional to the covariance
            between the log-rate at t_k and at t_{k-1}.
        sigma2: covariance function multiplier. Is also the marginal
            variance.
        mu: the means of the rates (not logged) for PX0. Of shape
          (n*(n-1), ) or (n, n).
        y_init: initial configuration of RWs (set to None or don't pass 
          into initialisation if using the default).
         px_verbose: if True and if t % 20 == 0, prints the value of t in PX.

        ----- Notes -----
        - Track the list of rates rather than generator A.
        - lams: ndarray of shape (n*(n-1), ). All rates in a 1D array, ordered
          as 12, ..., 1n, 21, 23, ..., 2n, ..., n(n-2), n(n-1).
        - lams are now actually the log-rates.
        - y defined as in type 4.
        - y_init: ndarray of shape (J, ).
        - SSM starts with the RWs spread across the states as evenly as
          possible by default (y_init == None).
        - y_k (data[k]): ndarray of shape (1, J), by convention.
        - CTMC states are 0 to n-1 in the code itself, however. This is
          because dists.Categorical returns values between 0 and n-1.
        - the mean function m(t) is constant and is ln(mu) for all t.
    """
    
    def __init__(self, *,
            n: int,
            J: int,
            delta_t: float,
            kappa: float,
            sigma2: float,
            mu,
            y_init = None,
            px_verbose: bool = False,
            px_verbose_int: int = 20
        ):
        
        assert n >= 2 and J >= 1 and delta_t > 0
        self.n = n
        self.J = J
        self.delta_t = delta_t
        
        if kappa <= 0 or kappa > 1:
            raise ValueError(
                "kappa must be bigger than 0 and less than or equal to 1. "
                f"Currently, its value is {kappa}."
            )
        self.kappa = kappa
        
        if sigma2 <= 0:
            raise ValueError(
                f"sigma2 must be positive. Currently, its value is {sigma2}."
            )
        self.sigma2 = sigma2
        
        self.mu = mu
        if len(self.mu.shape) > 1: # Assuming currently a (n, n) ndarray
            self.mu = gen_to_lams(self.mu)
        self.lmu = np.log(self.mu)
        
        assert self.lmu.shape == (self.n * (self.n - 1), )
        self.num_lams = self.n * (self.n - 1)
        
        if y_init is None:
            self.y_init = get_default_rw_initial_config(self.n, self.J)
        else:
            self.y_init = y_init
        
        # Compute transition scale
        self.TS = np.sqrt( self.sigma2 * (1 - self.kappa ** 2) )
        
        # Compute marginal distribution scale
        self.sigma = np.sqrt(self.sigma2)
        
        # Compute l
        self.l = -self.delta_t / np.log(self.kappa)
        
        self.px_verbose = px_verbose
        self.px_verbose_int = px_verbose_int
    
    
    def PX0(self):
        return dists.IndepProd(
            *[dists.Normal(lmu_pq, self.sigma) for lmu_pq in self.lmu]
        )
    
    
    def PX(self, t, xp):
        
        if self.px_verbose and t % self.px_verbose_int == 0:
            print("t:", t)
        
        lams_dists = [
            dists.Normal(
                self.lmu[i] + self.kappa * (xp[:, i] - self.lmu[i]),
                self.TS
            )
            for i in range(xp.shape[1])
        ]
        
        return dists.IndepProd(*lams_dists)
    
    
    def PY(self, t, xp, x, datap=None):
        
        if datap is None: # t == 0
            datap = self.y_init
        else: # t >= 1, datap == data[t-1]
            # By convention, datap is originally of shape (1, J)
            # Reshape it to (J, )
            datap = datap.reshape(-1)
        
        exp_x = np.exp(x)
        P_mat = np.stack(
            [
                compute_transition_prob_matrix(
                    cur_lams,
                    self.n,
                    self.delta_t
                )
                for cur_lams in exp_x
            ],
            axis=0
        )
        
        return dists.IndepProd(
            *[get_cat_dist(P_mat, y_i) for y_i in datap]
        )


