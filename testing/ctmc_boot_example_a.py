# -*- coding: utf-8 -*-

""" Bootstrap PF """

## Required since ctmc_boot is not in root folder ##
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
    simulate_example_a, simulate_data
)

## FUNCTIONS ##

def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
    """
    if p == 0 and q == 1:
        return "$\\varepsilon$"
    elif p == 1 and q == 2:
        return "$\\delta$"
    elif p == 2 and q == 0:
        return "$\\varphi$"
    else:
        return f"$\\lambda^{{{p} \\to {q}}}$"


# %%

## CTMC SSM Parameters ##

delta_t = 0.01 # Time between observations
C = 1 # Transition variance parameter
n = 3 # Number of states in CTMC
J = 5 # Number of random walkers
mu0  = np.array([1, 1, 1, 1, 5, 1]) # For P(A_0), a vague prior
var0 = np.array([4, 4, 4, 4, 6, 4]) # For P(A_0), a vague prior

## Particle filtering, simulation and other parameters ##

N = 10000 # Number of particles in PFs

K = 300 # k = 0, ..., K
k_series = np.arange(K + 1) # [0, 1, ..., K]
time_points = delta_t * k_series # [t_0, t_1, ..., t_K]

# %%

## Gamma dist. params. for PX0 ##
a0, b0 = get_gamma_params_from_mean_var(mu0, var0)

## All RWs start in state 0 ##
y_init = np.array([0 for _ in range(J)])

## For a Bootstrap PF ##
ctmc_ssm = CTMC(n=n, J=J, delta_t=delta_t, C=C, a0=a0, b0=b0, y_init=y_init,
                px_var_flag=False, px_verbose=True)

# %%

## Example A true rates ##

true_states = simulate_example_a(K=K)
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

fk_boot = augssm.AugmentedBootstrap(ssm=ctmc_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
print("Beginning the bootstrap particle filter.")
pf_boot.run()
print("Bootstrap particle filter finished.")

## Store lambda particles and weights in an xarray.Dataset ##

ds_boot = xr.Dataset({
    'X': xr.DataArray(
        np.stack([pf_boot.hist.X[k] for k in k_series]),
        dims=("k", "particle", "lam"),
        coords={
            "k": k_series,
            "lam": true_lams.columns.values,
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
).assign_coords(quantile=qs)

# %%

## Band plots: 5th quantile, median, 95th quantile ##

for lam_idx, lam in enumerate(true_lams.columns):
    p, q = lams_idx_to_gen_pos(lam_idx, n)
    latex_symbol = get_latex_rate_symbol(p, q)
    
    median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
    lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
    uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    plt.plot(k_series, true_lams[lam].values, label=f"True {latex_symbol}",
             color='red', alpha=0.7)
    plt.plot(median, color="green",
             label="PF mean", alpha=0.7)
    plt.fill_between(k_series, 
                     y1=lq, 
                     y2=uq, 
                     color="green", alpha=0.3)
    plt.legend()
    plt.xlabel("k")
    plt.ylabel(f"Value of {latex_symbol}")
    plt.title(f"Boot PF band plot (quantiles): {latex_symbol} | "
              + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
    plt.show()

# %%

## ESS ##

plt.plot(k_series, pf_boot.summaries.ESSs, color="red")
plt.xlabel("k")
plt.ylabel("ESS")
plt.title("ESS over time: Boot PF | "
          + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
plt.show()

# %%

## Choose some k between 0 and K+1 (inclusive) ##

k = np.random.randint(K+1)

# %%

## KDE for each lam (uses weights of particles) ##

for lam in true_lams.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(x=ds_boot["X"].sel({'lam': lam, 'k': k}).values.reshape(-1),
                weights=ds_boot["W"].sel({'k': k}).values.reshape(-1),
                ax=ax, fill=True,
                color="skyblue", label="Boot")
    ax.axvline(x=true_lams.loc[k][lam], color='red', linestyle=':',
               linewidth=1.5, label='True state')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Boot Filtering Dist. @ k = {k}: {lam}")
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# %%

## Pairwise scatter plots of particles (not using weights) ##

plot_df = (
    ds_boot["X"].sel(k=k)
    .to_pandas() # index: particle, columns: lambda
    .reset_index(drop=True)
)

sns.pairplot(
    plot_df,
    plot_kws={"alpha": 0.5, "s": 15}
)
plt.suptitle(f"Pairwise scatter at k = {k}: Boot PF", y=1.02)
plt.show()

# %%

## Band plots: bad way ##

# means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
# vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

# for lam_idx, lam in enumerate(true_lams.columns):
#     plt.plot(k_series, true_lams[lam].values, label=f"True {lam}",
#              color='red', alpha=0.7)
#     plt.plot(means_boot[..., lam_idx], color="green",
#              label="PF mean", alpha=0.7)
#     plt.fill_between(k_series, 
#                      y1=(means_boot[..., lam_idx]
#                          -2*np.sqrt(vars_boot[..., lam_idx])), 
#                      y2=(means_boot[..., lam_idx]
#                          +2*np.sqrt(vars_boot[..., lam_idx])), 
#                      color="green", alpha=0.3)
#     plt.legend()
#     plt.xlabel("k")
#     plt.ylabel(f"Value of {lam}")
#     p, q = lams_idx_to_gen_pos(lam_idx, n)
#     plt.title(f"Boot PF band plot (bad way): $\\lambda^{{{p} \\to {q}}}$ | "
#               + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
#     plt.show()
