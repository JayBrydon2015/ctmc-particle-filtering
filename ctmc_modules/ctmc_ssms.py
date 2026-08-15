# %%

# -*- coding: utf-8 -*-

""" CTMC SSMs as Python objects (classes). """

## IMPORTS ##

import numpy as np
from scipy.linalg import expm # For computing the matrix exponential of A: e^A

from particles import augmented_state_space_models as augssm
from particles import distributions as dists


## CONSANTS ##

GAMMA_TV_TYPES = ["PV", "CV"]


## FUNCTIONS ##

def get_gamma_params_from_mean_var(mean, var):
    """ Compute the Gamma distribution parameters, alpha
        and beta, from the mean and variance.
    """
    return mean ** 2 / var, mean / var

# lams are assumed to be a list of rates ordered by
# 12, 13, ..., 1n, 21, 23, ..., 2n, ..., n(n-1)
def gen_to_lams(gen):
    """ Convert generator A to lams.
        Essentially flattens and removes the diagonal elements.
        Can also be used to convert Gamma dist. parameters if they're
        of the same shape as the generator.
    """
    n = gen.shape[0]
    lams = np.array([])
    for m in range(n):
      lams = np.append(lams, gen[m,0:m])
      lams = np.append(lams, gen[m,m+1:])
    return lams

def lams_to_gen(lams):
    """ Convert lams to generator A. """
    l = len(lams)
    n = int((1 + np.sqrt(1 + 4*l))/2)
    gen = np.zeros((n,n))
    for m in range(n):
      lams_m = lams[m*(n-1):(m+1)*(n-1)]
      gen[m, 0:m] = lams_m[0:m]
      gen[m, m+1:] = lams_m[m:]
      gen[m,m] = - np.sum(lams_m)
    return gen

def lams_idx_to_gen_pos(idx, n):
    """ Returns the p, q position of the lambda given its index
        in the lams array and the number of CMTC states n.
        1 is added to i & j to go from 0-based to 1-based indexing.
    """
    i = idx // (n-1)
    j = idx %  (n-1)
    if i <= j:
        j += 1
    return i + 1, j + 1

def gen_pos_to_lams_idx(p: int, q: int, n: int):
    """
    Given p, q, and n, return the index in the flattened array

        L_1_2, L_1_3, ..., L_1_n,
        L_2_1, L_2_3, ..., L_2_n,
        ...
        L_n_1, ..., L_n_(n-1).

    States p and q are numbered from 1 to n.
    """

    if not (1 <= p <= n and 1 <= q <= n):
        raise ValueError(
            f"p and q must both be between 1 and {n}. "
            f"Got p={p}, q={q}."
        )

    if p == q:
        raise ValueError(
            "p and q must be different because self-transition "
            "rates are not included."
        )

    # Starting index of the block corresponding to p
    lam_idx = (p - 1) * (n - 1)

    # Position of q within that block, accounting for the missing p -> p
    if q < p:
        lam_idx += q - 1
    else:
        lam_idx += q - 2

    return lam_idx

def get_default_rw_initial_config(n: int, J: int):
    """ The random walkers are initialised approximately uniformly around
        the continuous-time Markov chain.
    """
    return np.sort(
        np.array([i % n for i in range(J)])
    )

def get_rw_all_state_p_initial_config(*, J: int, p: int = 1, n: int = None):
    """ The random walkers are initialised to all start in state p. """
    if n is not None and not 1 <= p <= n:
        raise ValueError("p must be between 1 and n (inclusive).")
    return np.array([(p - 1) for _ in range(J)])

def compute_transition_prob_matrix(lams, n, delta_t):
    """ Computes the matrix exponential of delta_t * lams_to_gen(lams). """
    return expm(delta_t * lams_to_gen(lams))

def get_cat_dist(P_mat, y_i):
    return dists.Categorical(P_mat[:, y_i])

def compute_weighted_mean_var(x, lw):
    """ Computes the weighted mean and variance along axis 0 of x.
        lw is the log-weights, which need to be transformed and normalised.
        
        --- Inputs ---
        x: ndarray of shape (N, L)
        lw: ndarray of shape (N,)
        
        --- Outputs ---
        mean: ndarray of shape (L,)
        var: ndarray of shape (L,)
        
        Returns the tuple (mean, var)
    """
    lw[np.isnan(lw)] = -np.inf
    m = lw.max()
    w = np.exp(lw - m)
    s = w.sum()
    W = w / s
    mean = (W[:, None] * x).sum(axis=0)
    var = (W[:, None] * (x - mean)**2).sum(axis=0)
    return mean, var

def kernel_se(t1, t2, l):
    """ Squared exponential kernel. """
    return np.exp( - (t1 - t2) ** 2 / (2 * l ** 2) )



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
            tv_type: str,
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
        
        if tv_type not in GAMMA_TV_TYPES:
            raise ValueError(
                f"The tv_type must be one of {GAMMA_TV_TYPES}. "
                f"Currently, tv_type is {tv_type}."
            )
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





###### CODE NOT USED ######

# class CTMC_prop(CTMC):
#     """ CTMC SSM with proposal.
    
#         ----- New parameters -----
        
#         Np: number of temporary particles used to calculate proposal
#         parameters for proposal0. Keep large enough (at least in the
#         hundreds, say). For proposal, the number of temporary particles
#         sampled using PX is the actual number of particles N (xp.shape[0]).
        
#         kappa: the balance between using the bootstrap filtering parameters
#         and the usual parameters in self.PX & self.PX0.
        
#         ! NOT USED !
#     """
    
#     def __init__(self, *, n, J, delta_t, C, a0, b0, y_init=None,
#                  Np=1000, kappa=0.5, kappa0=0.8):
#         self.n = n
#         self.J = J
#         self.delta_t = delta_t
#         self.C = C
#         self.a0 = a0
#         self.b0 = b0
#         if len(self.a0.shape) > 1: # Assuming currently a (n, n) ndarray
#             self.a0 = gen_to_lams(self.a0)
#             self.b0 = gen_to_lams(self.b0)
#         assert (
#             self.a0.shape[0] == self.n * (self.n - 1) and
#             self.a0.shape    == self.b0.shape
#         )
#         self.num_lams = self.n * (self.n - 1)
#         if y_init is None:
#             self.y_init = np.sort(np.array([i % self.n
#                                             for i in range(self.J)]))
#         else:
#             self.y_init = y_init
#         self.Np = Np
#         self.kappa0 = kappa0
#         if self.kappa0 < 0 or self.kappa0 > 1:
#             raise ValueError("kappa0 needs to be between 0 and 1 (inclusive).")
#         self.kappa = kappa
#         if self.kappa < 0 or self.kappa > 1:
#             raise ValueError("kappa needs to be between 0 and 1 (inclusive).")
    
#     def proposal0(self, data):
#         x_temp = self.PX0().rvs(size=self.Np)
#         lw_temp = self.PY(0, None, x_temp).logpdf(data[0])
#         mean, var = compute_weighted_mean_var(x_temp, lw_temp)
        
#         new_mean = self.kappa0 * self.a0 / self.b0 + (1 - self.kappa0) * mean
#         new_var = ( self.kappa0 * self.a0 / self.b0 ** 2
#                     + (1 - self.kappa0) * var )
        
#         alpha, beta = get_gamma_params_from_mean_var(new_mean, new_var)
#         lams_dists = [dists.Gamma(a, b) for a, b in zip(alpha, beta)]
#         return dists.IndepProd(*lams_dists)
    
#     def proposal(self, t, xp, data):
#         x_temp = self.PX(t, xp).rvs(size=xp.shape[0])
#         lw_temp = self.PY(t, xp, x_temp, data[t-1]).logpdf(data[t])
#         mean, var = compute_weighted_mean_var(x_temp, lw_temp)
        
#         # print(f"t: {t} | mean: {mean} | var: {var}")
#         # if np.isnan(mean).any():
#         #     print()
#         #     print(x_temp)
#         #     print()
#         #     print(lw_temp)
#         #     raise ValueError("nan encountered.")
        
#         if np.isnan(mean).any() or np.isnan(var).any():
#             return self.PX(t, xp)
        
#         new_mean = self.kappa * xp + ( 1 - self.kappa) * mean
#         new_var = ( self.kappa * self.C * self.delta_t * xp
#                    + (1 - self.kappa) * var)
        
#         alpha, beta = get_gamma_params_from_mean_var(new_mean, new_var)
#         lams_dists = [dists.Gamma(alpha[:, l], beta[:, l])
#                       for l in range(alpha.shape[1])]
#         return dists.IndepProd(*lams_dists)

