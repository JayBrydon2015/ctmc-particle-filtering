# -*- coding: utf-8 -*-

"""

Functions and helper functions.

"""


#### IMPORTS ####

import numpy as np
import xarray as xr
import pandas as pd
from scipy.linalg import expm # Matrix exponential of A: e^A

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments
from particles import distributions as dists


#### CTMC, SSM, and PF FUNCTIONS ####


def get_gamma_params_from_mean_var(mean, var):
    """ Compute the Gamma distribution parameters, alpha
        and beta, from the mean and variance.
    """
    return mean ** 2 / var, mean / var


def gen_to_lams(gen):
    """ Convert generator A to lams.
        Essentially flattens and removes the diagonal elements.
        Can also be used to convert Gamma dist. parameters if they're
        of the same shape as the generator.
        
        lams are assumed to be a list of rates ordered by
        12, 13, ..., 1n, 21, 23, ..., 2n, ..., n(n-1).
    """
    n = gen.shape[0]
    lams = np.array([])
    for m in range(n):
      lams = np.append(lams, gen[m,0:m])
      lams = np.append(lams, gen[m,m+1:])
    return lams


def lams_to_gen(lams):
    """ Convert lams to generator A.
        
        lams are assumed to be a list of rates ordered by
        12, 13, ..., 1n, 21, 23, ..., 2n, ..., n(n-1).
    """
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
    """ Given p, q, and n, return the index in the flattened array
        
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


def compute_transition_prob_matrix(lams, n: int, delta_t: float):
    """ Computes the matrix exponential of delta_t * lams_to_gen(lams). """
    return expm(delta_t * lams_to_gen(lams))


def get_cat_dist(P_mat, y_i):
    """ Returns the categorical distribution for a particular
        random walker (and for every particle).
    """
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


def weighted_quantile(values, weights, quantiles):
    """ Calculates weighted quantiles using linear interpolation.
    
        The weighted empirical cumulative distribution function is constructed
        from the values and weights, and the requested quantiles are obtained
        by linearly interpolating its inverse.
    
        -- Inputs --
        values: (N,)
            Values for which to calculate the weighted quantiles.
        weights: (N,)
            Corresponding non-negative weights.
        quantiles: array-like in [0, 1]
            Quantiles to calculate.
    """
    
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)


def run_particle_filter(*,
        ctmc_ssm,
        data,
        N,
        k_series,
        exp_X: bool = False,
        qs = None
    ):
    
    """
    Runs the bootstrap particle filter and stores the results
    into an xarray.Dataset.
        
    If quantiles is None, calculate and store the quantiles 5%, 50%, and
    95% of the filtering ensembles in the Dataset.
        
    Inputs
    ------
    ctmc_ssm: continuous-time Markov chain state-space model as a Python
        object.
    data: a list of length K+1 containing the data/observations.
    N: number of particles in the particle filter.
    lam_names: ["L_1_2", "L_1_3", ..., f"L_{n_{n-1}"], where f"L_{p}_{q}"
        refers to the transition rate from states p to q and n is the number
        of states in the continuous-time Markov chain. Note that p != q.
        See docstring in make_true_rates_dataframe for more details.
    k_series: [0, 1, 2, ..., K-1, K], a list, NumPy 1D ndarray, etc.
    exp_X: if True, exponentiate the particles' values. Done so with the
        Gaussian process runs.
    qs: NumPy ndarray of shape (Q,) containing Q quantiles to compute, each
        between 0 and 1.
    """
    
    if qs is None:
        qs = np.array([0.05, 0.5, 0.95]) # 90% interval & median
    
    
    #### Run the bootstrap particle filter ####
    
    fk_boot = augssm.AugmentedBootstrap(ssm=ctmc_ssm, data=data)
    pf_boot = particles.SMC(
        fk=fk_boot, N=N,
        resampling='stratified', 
        store_history=True, collect=[Moments()]
    )
    print("Beginning the bootstrap particle filter.")
    pf_boot.run()
    print("Bootstrap particle filter finished.")
    print()
    
    
    #### Store lambda particles and weights in an xarray.Dataset ####
    
    lams_gen_positions = [
        lams_idx_to_gen_pos(i, ctmc_ssm.n)
        for i in range(ctmc_ssm.num_lams)
    ]
    lam_names = [f"L_{p}_{q}" for p, q in lams_gen_positions]
    
    ds_boot = xr.Dataset({
        
        'X': xr.DataArray(
            np.stack([pf_boot.hist.X[k] for k in k_series]),
            dims=("k", "particle", "lam"),
            coords={
                "k": k_series,
                "lam": lam_names,
            },
            name="Bootstrap PF Particles"
        ),
        
        'W': xr.DataArray(
            np.stack([pf_boot.hist.wgts[k].W for k in k_series]),
            dims=("k", "weight"),
            coords={
                "k": k_series
            },
            name="Bootstrap PF Weights"
        ),
        
        'ESS': xr.DataArray(
            pf_boot.summaries.ESSs,
            dims="k",
            coords={
                "k": k_series
            },
            name="Bootstrap PF ESSs"
        )
    })
    
    # If exp_X, exponentiate the log-rates
    if exp_X:
        ds_boot["X"] = np.exp(ds_boot["X"])
    
    
    #### Calculate quantiles and add into ds_boot ####
    
    ds_boot["X_quantiles"] = xr.apply_ufunc(
        weighted_quantile,
        ds_boot["X"],
        ds_boot["W"],
        input_core_dims=[["particle"], ["weight"]],
        output_core_dims=[["quantile"]],
        vectorize=True,
        kwargs={"quantiles": qs},
        dask="parallelized",
        output_dtypes=[float],
    ).assign_coords(quantile=qs).rename("Bootstrap PF Quantiles")
    
    return ds_boot


def get_p_q_from_lam_col_name(lam_col: str):
    """ Given L_p_q, returns p and q as integers. """
    p, q = lam_col.split("_")[1:]
    return int(p), int(q)


def make_true_rates_dataframe(*, true_rates, n):
    """
    Creates a pandas data frame storing the true transition rates. The columns
    of the data frame refer to each of the transition rates and each column's
    name is 'L_p_q', referring to the transition rate from states p to q.
    
    Assumes the rates are ordered as L_1_2, L_1_3, ..., L_1_n, L_2_1,
    L_2_3, ..., L_2_n, ..., L_n_1, L_n_2, ..., L_n_n-1, where L_p_q refers
    to the transition rate from states p to q. Note that p and q are between
    1 and n, and p != q.
    
    Inputs
    ------
    true_rates: a list containing the true transition rates at each time in
        time_series. true_rates[i] are the true transition rates at i and
        true_rates[i] is a NumPy ndarray of shape (1, n*(n-1)), where n*(n-1)
        is the maximum number of transition rates in a continuous-time
        Markov chain with n states.
    n: number of states in the continuous-time Markov chain.
    """
    
    num_rates = n * (n - 1)
    
    lams_gen_positions = [
        lams_idx_to_gen_pos(i, n)
        for i in range(num_rates)
    ]
    
    true_rates_df = pd.DataFrame(
        np.stack([true_rates_i.reshape(-1) for true_rates_i in true_rates]),
        columns=[f"L_{p}_{q}" for p, q in lams_gen_positions]
    )
    
    return true_rates_df


#### PLOTTING FUNCTIONS ####


def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate, L_p_q, that could be
        used in the image file's name, for example.
    """
    return f"$\\lambda^{{{p} \\to {q}}}$", f"L_{p}_{q}"


def map_num_to_letter(num: int) -> str:
    """ Maps 0 to 'A', 1 to 'B', 2 to 'C', up to 25 to 'Z'. """
    if 0 <= num <= 25:
        return chr(65 + num)
    raise ValueError("Input must be between 0 and 25 inclusive.")


def apply_font_sizes(fig, font_size, tick_font_size):
    """ Given fig, change all its font sizes according to
        font_size and tick_font_size.
    """
    for ax in fig.axes:
        ax.title.set_fontsize(font_size)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis="both", labelsize=tick_font_size)

        ax.xaxis.get_offset_text().set_fontsize(tick_font_size)
        ax.yaxis.get_offset_text().set_fontsize(tick_font_size)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(font_size)
            legend.get_title().set_fontsize(font_size)


