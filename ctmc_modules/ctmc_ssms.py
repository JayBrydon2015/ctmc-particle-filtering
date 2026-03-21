# %%

# -*- coding: utf-8 -*-

""" PF for inferring rates of CTMC """

## Imports ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from scipy.special import gammaln
from scipy.linalg import expm # For computing the matrix exp: e^A

# Particles package
import particles
from particles import augmented_state_space_models as augssm
from particles import distributions as dists


## Functions ##

def sigmoid(x, a, b, m, k):
    """ Result = a when t = 0. """
    result = a + (b-a) / (1 + np.exp(-k*(x-m)))
    result -= a + (b-a) / (1 + np.exp(k * m))
    result += a
    return result

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


## CTMC SSM Classes ##


class CTMC(augssm.AugmentedStateSpaceModel):
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
        - y: ndarray of shape (n*J, )
        - lams: ndarray of shape (n*(n-1), )
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
        # return dists.Categorical(P_mat[:, y_i])

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


class CTMC_prop(CTMC):
    """ CTMC SSM with proposal.
    
        CURRENTLY NOT WORKING!
    """
    
    def proposal0(self, data):
        return self.PX0()
    
    def compute_transition_count(self, datap, data):
        return np.bincount(self.n * datap.reshape(-1) + data.reshape(-1),
                minlength=self.n * self.n).reshape(self.n, self.n)
    
    def compute_numerator_or_denominator(self, mu, n, a, b, m):
        return ((-1) ** m * comb(a, m) *
                (1 / self.delta_t + self.delta_t * (b + m)) **
                (-mu / self.delta_t - n))
    
    def get_nth_moment(self, mu, n, a, b):
        numerator_result = sum(
            self.compute_numerator_or_denominator(mu, n, a, b, m)
            for m in range(a+1)
        )
        denominator_result = sum(
            self.compute_numerator_or_denominator(mu, 0, a, b, m)
            for m in range(a+1)
        )
        R = np.exp( gammaln(n + mu / self.delta_t) - gammaln(mu / self.delta_t) )
        result = R * numerator_result / denominator_result
        # print(f"a: {a}")
        # print(f"b: {b}")
        # print(f"numerator: {numerator_result}")
        # print(f"denominator: {denominator_result}")
        # print(f"result: {result}")
        return result
    
    def proposal(self, t, xp, data):
        lams_means = np.mean(xp, axis=0)
        trans_count_mat = self.compute_transition_count(data[t-1], data[t])
        lams_dists = []
        for idx, mu in enumerate(lams_means):
            p, q = lams_idx_to_gen_pos(idx, self.n)
            # Compute a & b
            a = trans_count_mat[p, q]
            b = trans_count_mat[p, p]
            # Compute first and second moments
            # print(f"idx: {idx}")
            # print(f"mu: {mu}")
            if np.isnan(mu):
                print(xp[:, idx])
                raise ValueError("mu is np.nan!")
            first_mom  = self.get_nth_moment(mu, 1, a, b)
            second_mom = self.get_nth_moment(mu, 2, a, b)
            var = second_mom - first_mom ** 2
            alpha, beta = get_gamma_params_from_mean_var(first_mom, var)
            lams_dists.append(dists.Gamma(alpha, beta))
        return dists.IndepProd(*lams_dists)

