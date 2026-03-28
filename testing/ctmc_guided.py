# -*- coding: utf-8 -*-

""" Guided PF """

## Required since ctmc_guided is not in root folder ##
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parents[0]))

## Imports ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr

from ctmc_modules.ctmc_ssms import (
    get_gamma_params_from_mean_var,
    gen_to_lams,
    lams_to_gen,
    lams_idx_to_gen_pos,
    compute_transition_prob_matrix,
    CTMC,
    CTMC_prop,
)

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments

from rates_simulation.true_rates_simulation_funtions import (
    simulate_sigmoid_growth, simulate_constant_rates, simulate_data
)

# %%

## CTMC SSM Parameters ##

delta_t = 0.02 # Time between observations
C = 1 # Transition variance parameter
n = 2 # Number of states in CTMC
J = 5 # Number of random walkers
mu0  = np.array([2, 3]) # For P(A_0)
var0 = np.array([0.2, 0.2]) # For P(A_0)

## Particle filtering, simulation and other parameters ##

N = 1000 # Number of particles in PFs

K = 300 # k = 0, ..., K
k_series = np.arange(K + 1) # [0, 1, ..., K]
time_points = delta_t * k_series # [t_0, t_1, ..., t_K]

# %%

## Gamma dist. params. for PX0 ##
a0, b0 = get_gamma_params_from_mean_var(mu0, var0)

## For a Guided PF ##
ctmc_ssm = CTMC_prop(n=n, J=J, delta_t=delta_t, C=C, a0=a0, b0=b0)

## Get y_init from ctmc_ssm object ##
y_init = ctmc_ssm.y_init

# %%

## Sigmoid growth of true rates ##

max_growth = np.array([2, 1]) # Must be same shape as mu0
sig_func_val = 14 # Tune this to achieve desired shape of sigmoid growth
true_states = simulate_sigmoid_growth(mu0=mu0, max_growth=max_growth, K=K,
                                      sig_func_val=sig_func_val)
data = simulate_data(true_rates=true_states, n=n, J=J, delta_t=delta_t,
                     y_init=y_init)

## Store true lambdas in Pandas dataframe ##

lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                      for i in range(true_states[0].shape[1])]
true_lams = pd.DataFrame(np.stack([true_state.reshape(-1)
                                   for true_state in true_states]),
                         columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
                         index=k_series)
true_lams = true_lams.rename_axis('k')

## Plot true rates ##

for col in true_lams.columns:
    plt.plot(k_series, true_lams[col], label=col)
plt.xlabel("k")
plt.ylabel("Value")
plt.title("True rates over time")
plt.legend()
plt.tight_layout()
plt.show()

# %%

## Plot data (if J not too large) ##

if J <= 10:
    data_plot = np.vstack(data)
    
    fig, axes = plt.subplots(
        nrows=J,
        ncols=1,
        sharex=True,
        figsize=(8, 4 * n)
    )
    fig.suptitle("RW states over time", fontsize=14)
    
    # Ensure axes is always iterable (important if J == 1)
    if J == 1:
        axes = [axes]
    
    for j in range(J):
        axes[j].plot(k_series, data_plot[:, j])
        axes[j].set_ylabel(f"RW #{j+1}")
        axes[j].grid(True)
    
    axes[-1].set_xlabel("k")
    plt.tight_layout()
    plt.show()
else:
    print(f"Too many random walkers to plot: {J} RWs.")

# %%

## Run the bootstrap PF ##

fk_guided = augssm.AugmentedGuidedPF(ssm=ctmc_ssm, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
print("Beginning the guided particle filter.")
pf_guided.run()
print("Guided particle filter finished.")

## Store lambda particles and weights in an xarray.Dataset ##

ds_guided = xr.Dataset({
    'X': xr.DataArray(
        np.stack([pf_guided.hist.X[k] for k in k_series]),
        dims=("k", "particle", "lam"),
        coords={
            "k": k_series,
            "lam": true_lams.columns.values,
        },
        name="Guided PF Particles"
    ),
    'W': xr.DataArray(
        np.stack([pf_guided.hist.wgts[k].W for k in k_series]),
        dims=("k", "weight"),
        coords={
            "k": k_series
        },
        name="Guided PF Weights"
    )
})

## Calculate quantiles and add into ds_boot ##

def weighted_quantile(values, weights, quantiles):
    """
    values: (particle,)
    weights: (particle,)
    quantiles: array-like in [0,1]
    """
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)

qs = np.array([0.05, 0.5, 0.95]) # 95% interval & median

ds_guided["X_quantiles"] = xr.apply_ufunc(
    weighted_quantile,
    ds_guided["X"],
    ds_guided["W"],
    input_core_dims=[["particle"], ["weight"]],
    output_core_dims=[["quantile"]],
    vectorize=True,
    kwargs={"quantiles": qs},
    dask="parallelized",
    output_dtypes=[float],
).assign_coords(quantile=qs)

# %%

## Band plots: 5th quantile, median, 95th quantile ##

for lam_idx, lam in enumerate(true_lams.columns):
    median = ds_guided["X_quantiles"].sel(lam=lam, quantile=0.5)
    lq = ds_guided["X_quantiles"].sel(lam=lam, quantile=0.05)
    uq = ds_guided["X_quantiles"].sel(lam=lam, quantile=0.95)
    plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
             color='red', alpha=0.7)
    plt.plot(median, color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(k_series, 
                     y1=lq, 
                     y2=uq, 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("k")
    plt.ylabel(f"Value of {lam}")
    p, q = lams_idx_to_gen_pos(lam_idx, n)
    plt.title("Guided PF band plot (quantiles): "
              + f"$\\lambda^{{{p} \\to {q}}}$ | "
              + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
    plt.show()

# %%

## ESS ##

plt.plot(k_series, pf_guided.summaries.ESSs, color="red")
plt.xlabel("k")
plt.ylabel("ESS")
plt.title("ESS over time: Guided PF | "
          + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
plt.show()

# %%

## Choose some k between 0 and K+1 (inclusive) ##

k = np.random.randint(K+1)

# %%

## KDE for each lam (uses weights of particles) ##

for lam in true_lams.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(x=ds_guided["X"].sel({'lam': lam, 'k': k}).values.reshape(-1),
                weights=ds_guided["W"].sel({'k': k}).values.reshape(-1),
                ax=ax, fill=True,
                color="skyblue", label="Boot")
    ax.axvline(x=true_lams.loc[k][lam], color='red', linestyle=':',
               linewidth=1.5, label='True state')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Guided Filtering Dist. @ k = {k}: {lam}")
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# %%

## Pairwise scatter plots of particles (not using weights) ##

plot_df = (
    ds_guided["X"].sel(k=k)
    .to_pandas() # index: particle, columns: lambda
    .reset_index(drop=True)
)

sns.pairplot(
    plot_df,
    plot_kws={"alpha": 0.5, "s": 15}
)
plt.suptitle(f"Pairwise scatter at k = {k}: Guided PF", y=1.02)
plt.show()

# %%

## Band plots: bad way ##

# means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
# vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

# for lam_idx, lam in enumerate(true_lams.columns):
#     plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
#              color='red', alpha=0.7)
#     plt.plot(means_guided[..., lam_idx], color="green",
#              label="PF mean", alpha=0.7)
#     plt.fill_between(k_series, 
#                      y1=(means_guided[..., lam_idx]
#                          -2*np.sqrt(vars_guided[..., lam_idx])), 
#                      y2=(means_guided[..., lam_idx]
#                          +2*np.sqrt(vars_guided[..., lam_idx])), 
#                      color="green", alpha=0.3)
#     plt.legend()
#     plt.xlabel("k")
#     plt.ylabel(f"Value of {lam}")
#     p, q = lams_idx_to_gen_pos(lam_idx, n)
#     plt.title("Guided PF band plot (bad way): "
#               + f"$\\lambda^{{{p} \\to {q}}}$ | "
#               + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
#     plt.show()
