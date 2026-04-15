# %%

# -*- coding: utf-8 -*-

""" CTMC SSMs as Python objects (classes). """

## Imports ##

import numpy as np
from scipy.linalg import expm # For computing the matrix exp: e^A

# Particles package
from particles import augmented_state_space_models as augssm
from particles import distributions as dists


## Functions ##

def get_gamma_params_from_mean_var(mean, var):
    """ Compute the Gamma distribution parameters, alpha
        and beta, from the mean and variance.
    """
    return mean ** 2 / var, mean / var

# lams are assumed to be a list of rates ordered by
# 11, 12, 13, ..., 1n, 21, 23, ..., 2n, ..., (n-1)n
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
        in the lams array and the number of CMTC states n. """
    i = idx // (n-1)
    j = idx %  (n-1)
    if i <= j:
        j += 1
    return i, j

def compute_transition_prob_matrix(lams, n, delta_t):
    ## Option 1: worse ##
    # return np.identity(n) + DELTA_T * lams_to_gen(lams)
    
    ## Option 2: better ##
    return expm(delta_t * lams_to_gen(lams))

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


## CTMC SSM Classes ##


class CTMC_old(augssm.AugmentedStateSpaceModel):
    """ CTMC Augmented SSM

        ----- Parameters -----
        n: number of states
        J: number of random walkers
        a0: Gamma dist alpha parameters for PX0 in (n, n) ndarray
        b0: Gamma dist beta parameters for PX0 in (n, n) ndarray

        ----- Notes -----
        - Track lams_list rather than generator A
        - y defined as in type 4
        - SSM starts with the RWs spread across the states as evenly as
          possible.
        - y: ndarray of shape (J, )
        - lams: ndarray of shape (n*(n-1), )
        
        BUGGED, don't use this.
    """

    def PX0(self):
        if len(self.a0.shape) > 1: # Currently a (n, n) ndarray
            self.a0 = gen_to_lams(self.a0)
            self.b0 = gen_to_lams(self.b0)
        lams_dists = [dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        return dists.IndepProd(*lams_dists)

    def PX(self, t, xp):
        ## Option 1: Var = lambda * DELTA_T * C ##
        
        alpha = xp / self.delta_t
        beta_i = np.repeat(1/self.delta_t, alpha.shape[0])
        alpha  /= self.C
        beta_i /= self.C
        lams_dists = [dists.Gamma(alpha[:, l], beta_i)
                      for l in range(alpha.shape[1])]
        
        ## Option 2: Var = DELTA_T * C ##
        
        # alpha = xp["lams"] ** 2 / self.delta_t
        # beta = xp["lams"] / self.delta_t
        # alpha /= self.C
        # beta  /= self.C
        # lams_dists = [dists.Gamma(alpha[:,l], beta[:, l])
        #               for l in range(alpha.shape[1])]
        
        return dists.IndepProd(*lams_dists)

    def get_cat_dist(self, P_mat, y_i):
        return dists.Categorical(P_mat[np.arange(P_mat.shape[0]), y_i])

    def PY(self, t, xp, x, datap=None):
        ## y ##
        
        if t == 0:
            y0 = np.sort(np.array([i % self.n for i in range(self.J)]))
            y0 = np.stack([y0 for _ in range(x.shape[0])], axis=0)
            y0_dists = [dists.DiscreteDirac(y0[..., r])
                        for r in range(self.J)]
            return dists.IndepProd(*y0_dists)
        
        else: # t >= 1
            P_mat = np.stack([compute_transition_prob_matrix(cur_lams, self.n,
                                                             self.delta_t)
                              for cur_lams in xp], axis=0)
            y_dists = [self.get_cat_dist(P_mat, datap[:, i])
                                for i in range(datap.shape[1])]
            return dists.IndepProd(*y_dists)


class CTMC(augssm.AugmentedStateSpaceModel):
    """ CTMC Augmented SSM

        ----- Parameters -----
        n: number of states
        J: number of random walkers
        delta_t: real time between observations
        C: to scale the transition variance by
        a0: Gamma dist alpha parameters for PX0 in (n, n) ndarray
        b0: Gamma dist beta parameters for PX0 in (n, n) ndarray
        y_init: initial configuration of RWs (set to None or don't pass 
          into initialisation if using the default)
        px_var_flag: decides which transition variance to use (see PX). If
          False (the default), the variance of \lambda_{k+1} | \lambda_{k} is
          \lambda_{k} * DELTA_T * C; if True, it is DELTA_T * C.
        reg_term: a small constant added to beta in Option 2 in PX to help
          numberical stability.
         px_verbose: if True and if t % 20 == 0, prints the value of t in PX.

        ----- Notes -----
        - Track lams_list rather than generator A
        - y defined as in type 4
        - SSM starts with the RWs spread across the states as evenly as
          possible by default (y_init == None)
        - y (data[k]): ndarray of shape (1, J), by convention
        - lams: ndarray of shape (n*(n-1), )
        - y_init: ndarray of shape (J, )
    """
    
    def __init__(self, *, n, J, delta_t, C, a0, b0, y_init=None,
                 px_var_flag=False, reg_term=1e-6, px_verbose=False):
        self.n = n
        self.J = J
        self.delta_t = delta_t
        self.C = C
        self.a0 = a0
        self.b0 = b0
        if len(self.a0.shape) > 1: # Currently a (n, n) ndarray
            self.a0 = gen_to_lams(self.a0)
            self.b0 = gen_to_lams(self.b0)
        if y_init is None:
            self.y_init = np.sort(np.array([i % self.n
                                            for i in range(self.J)]))
        else:
            self.y_init = y_init
        self.px_var_flag = px_var_flag
        self.reg_term = reg_term
        self.px_verbose = px_verbose

    def PX0(self):
        lams_dists = [dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        return dists.IndepProd(*lams_dists)
    
    def PX(self, t, xp):
        if self.px_verbose and t % 20 == 0:
            print("t:", t)
        Dt_times_C = self.delta_t * self.C
        if not self.px_var_flag:
            ## Option 1 (default): Var = lambda * DELTA_T * C ##
            alpha = xp / Dt_times_C
            beta_i = np.repeat(1 / Dt_times_C, alpha.shape[0])
            lams_dists = [dists.Gamma(alpha[:, l], beta_i)
                          for l in range(alpha.shape[1])]
        else:
            ## Option 2: Var = DELTA_T * C ##
            alpha = xp ** 2 / Dt_times_C
            beta  = xp / Dt_times_C + self.reg_term
            lams_dists = [dists.Gamma(alpha[:,l], beta[:, l])
                          for l in range(alpha.shape[1])]
        return dists.IndepProd(*lams_dists)
    
    def get_cat_dist(self, P_mat, y_i):
        return dists.Categorical(P_mat[:, y_i])

    def PY(self, t, xp, x, datap=None):
        ## y ##
        
        if datap is None: # t == 0
            datap = self.y_init
        else: # t >= 1, datap == data[t-1]
            # By convention, datap is originally of shape (1, J)
            # Reshape it to (J, )
            datap = datap.reshape(-1)
        
        P_mat = np.stack([compute_transition_prob_matrix(cur_lams, self.n,
                                                         self.delta_t)
                          for cur_lams in x], axis=0)
        y_dists = [self.get_cat_dist(P_mat, y_i) for y_i in datap]
        return dists.IndepProd(*y_dists)


class CTMC_prop(CTMC):
    """ CTMC SSM with proposal.
    
        ----- New parameters -----
        
        Np: number of temporary particles used to calculate proposal
        parameters for proposal0. Keep large enough (at least in the
        hundreds, say). For proposal, the number of temporary particles
        sampled using PX is the actual number of particles N (xp.shape[0]).
        
        kappa: the balance between using the bootstrap filtering parameters
        and the usual parameters in self.PX & self.PX0.
    """
    
    def __init__(self, *, n, J, delta_t, C, a0, b0, y_init=None,
                 Np=1000, kappa=0.5, kappa0=0.8):
        self.n = n
        self.J = J
        self.delta_t = delta_t
        self.C = C
        self.a0 = a0
        self.b0 = b0
        if len(self.a0.shape) > 1: # Currently a (n, n) ndarray
            self.a0 = gen_to_lams(self.a0)
            self.b0 = gen_to_lams(self.b0)
        if y_init is None:
            self.y_init = np.sort(np.array([i % self.n
                                            for i in range(self.J)]))
        else:
            self.y_init = y_init
        self.Np = Np
        self.kappa0 = kappa0
        if self.kappa0 < 0 or self.kappa0 > 1:
            raise ValueError("kappa0 needs to be between 0 and 1 (inclusive).")
        self.kappa = kappa
        if self.kappa < 0 or self.kappa > 1:
            raise ValueError("kappa needs to be between 0 and 1 (inclusive).")
    
    def proposal0(self, data):
        x_temp = self.PX0().rvs(size=self.Np)
        lw_temp = self.PY(0, None, x_temp).logpdf(data[0])
        mean, var = compute_weighted_mean_var(x_temp, lw_temp)
        
        new_mean = self.kappa0 * self.a0 / self.b0 + (1 - self.kappa0) * mean
        new_var = ( self.kappa0 * self.a0 / self.b0 ** 2
                    + (1 - self.kappa0) * var )
        
        alpha, beta = get_gamma_params_from_mean_var(new_mean, new_var)
        lams_dists = [dists.Gamma(a, b) for a, b in zip(alpha, beta)]
        return dists.IndepProd(*lams_dists)
    
    def proposal(self, t, xp, data):
        x_temp = self.PX(t, xp).rvs(size=xp.shape[0])
        lw_temp = self.PY(t, xp, x_temp, data[t-1]).logpdf(data[t])
        mean, var = compute_weighted_mean_var(x_temp, lw_temp)
        
        # print(f"t: {t} | mean: {mean} | var: {var}")
        # if np.isnan(mean).any():
        #     print()
        #     print(x_temp)
        #     print()
        #     print(lw_temp)
        #     raise ValueError("nan encountered.")
        
        if np.isnan(mean).any() or np.isnan(var).any():
            return self.PX(t, xp)
        
        new_mean = self.kappa * xp + ( 1 - self.kappa) * mean
        new_var = ( self.kappa * self.C * self.delta_t * xp
                   + (1 - self.kappa) * var)
        
        alpha, beta = get_gamma_params_from_mean_var(new_mean, new_var)
        lams_dists = [dists.Gamma(alpha[:, l], beta[:, l])
                      for l in range(alpha.shape[1])]
        return dists.IndepProd(*lams_dists)
