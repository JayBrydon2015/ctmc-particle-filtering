# -*- coding: utf-8 -*-

""" Bootstrap PF """


## Required since ctmc_boot is not in root folder ##

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parents[0]))


## CONSTANTS ##

PLOT_ROOT_FOLDER_NAME = "generated_plots"
EXAMPLE_FOLDER_NAME = "CTMC_ExampleA_Figs"
# Choose whether to show all plots, or save the plots for thesis
SAVE_PLOTS = False
PLOT_EXTRAS = False # Plot extra stuff


## Imports ##

import numpy as np
if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg') # Must be called before importing pyplot
    import matplotlib.pyplot as plt
else:
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
    GP_CTMC,
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
        Also returns the name of the rate used in the image file
        name.
    """
    if p == 0 and q == 1:
        return "$\\varepsilon$", "Eps"
    elif p == 1 and q == 2:
        return "$\\delta$", "Delta"
    elif p == 2 and q == 0:
        return "$\\varphi$", "Phi"
    else:
        return f"$\\lambda^{{{p} \\to {q}}}$", f"L{p}{q}"


## Plotting Config ##

# plt.rcParams.update({
#     'font.size': 14,
#     'axes.titlesize': 16,
#     'axes.labelsize': 14,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14
# })


# %%

## CTMC SSM Parameters ##

delta_t = 0.01 # Time between observations
l = 0.8 # SE Kernel parameter
n = 3 # Number of states in CTMC
J = 8 # Number of random walkers
mu0  = np.array([1.2, 0.5, 0.5, 0.9, 5, 0.5]) # For P(A_0)
scale0 = np.array([1, 1, 1, 1, 2, 1]) # For P(A_0)

## Particle filtering, simulation and other parameters ##

N = 10000 # Number of particles in PFs

K = 300 # k = 0, ..., K
k_series = np.arange(K + 1) # [0, 1, ..., K]
time_points = delta_t * k_series # [t_0, t_1, ..., t_K]


## Strings for directories for saving the plots ##
FOLDER_PATH_STR = (PLOT_ROOT_FOLDER_NAME + "/"
                   + f"{EXAMPLE_FOLDER_NAME}/GP; J={J}; l={l}; Dt={delta_t}")


# %%

## All RWs start in state 0 ##
y_init = np.array([0 for _ in range(J)])

## For a Bootstrap PF ##
ctmc_ssm = GP_CTMC(n=n, J=J, delta_t=delta_t, l=l, mu0=mu0, scale0=scale0,
                   y_init=y_init, px_verbose=True)

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

if PLOT_EXTRAS:
    for col in true_lams.columns:
        plt.plot(k_series, true_lams[col], label=col)
    plt.xlabel("k")
    plt.ylabel("Value")
    plt.title("True rates over time")
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.show()
        plt.close("all")
    else:
        plt.show()

# %%

## Plot data (if J not too large) ##

data_plot_fontsize = 23
data_plot_fontsize2 = 20

if J <= 8:
    data_plot = np.vstack(data)
    data_plot += 1 # '0' -> State 1, etc
    
    if J == 8:
        fig, axes = plt.subplots(
            nrows=4,
            ncols=2,
            sharex=True,
            figsize=(8, 8)
        )
        
        axes = axes.flatten()
        
        for j in range(J):
            axes[j].plot(k_series, data_plot[:, j], lw=2)
            # axes[j].set_ylabel(f"RW #{j+1}", fontsize=data_plot_fontsize)
            axes[j].tick_params(axis='both', labelsize=data_plot_fontsize)
            axes[j].grid(True)
            axes[j].set_ylim(0.8, 3.2)
            axes[j].set_xlim(-10, 310)
        
        for ax in axes[-2:]:
            ax.set_xlabel("k", fontsize=data_plot_fontsize)
        
        for ax in axes:
            ax.set_xticks([0, 100, 200, 300])
            ax.tick_params(axis='x', labelbottom=False)
    else:
        fig, axes = plt.subplots(
            nrows=J,
            ncols=1,
            sharex=True,
            figsize=(8, 4 * J)
        )
        
        # Ensure axes is always iterable (important if J == 1)
        if J == 1:
            axes = [axes]
        
        for j in range(J):
            axes[j].plot(k_series, data_plot[:, j], lw=2)
            # axes[j].set_ylabel(f"RW #{j+1}", fontsize=data_plot_fontsize2)
            axes[j].tick_params(axis='both', labelsize=data_plot_fontsize2)
            axes[j].grid(True)
            axes[j].set_ylim(0.8, 3.2)
            axes[j].set_xlim(-10, 310)
        
        axes[-1].set_xlabel("k", fontsize=data_plot_fontsize2)
    
    # fig.suptitle("RW states over time", fontsize=20)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        folder_path = Path(FOLDER_PATH_STR)
        folder_path.mkdir(parents=True, exist_ok=True)
        image_name = "rw_data.png"
        plt.savefig(folder_path / image_name,
                    bbox_inches='tight')
        plt.close("all")
    else:
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


## Exponentiate the log-rates ##

ds_boot["Exp_X"] = np.exp(ds_boot["X"])


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

ds_boot["Exp_X_quantiles"] = xr.apply_ufunc(
    weighted_quantile,
    ds_boot["Exp_X"],
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

BP_MIN_YLIMS = [-0.1, -0.1, -0.1, -0.1, 0.5, -0.1]
BP_MAX_YLIMS = [2, 2, 2, 2, 7, 2]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()  # makes indexing easy: 0..5

for lam_idx, lam in enumerate(true_lams.columns):
    ax = axes[lam_idx]

    p, q = lams_idx_to_gen_pos(lam_idx, n)
    latex_symbol, rate_name_img = get_latex_rate_symbol(p, q)
    
    median = ds_boot["Exp_X_quantiles"].sel(lam=lam, quantile=0.5)
    lq = ds_boot["Exp_X_quantiles"].sel(lam=lam, quantile=0.05)
    uq = ds_boot["Exp_X_quantiles"].sel(lam=lam, quantile=0.95)

    ax.plot(k_series, true_lams[lam].values,
            label="True rate",
            color='blue', alpha=0.8, lw=2)

    ax.plot(k_series, median,
            color="orange", label="PF median", alpha=0.7, lw=2)

    ax.fill_between(k_series,
                    y1=lq,
                    y2=uq,
                    color="orange", alpha=0.3, lw=2)

    ax.set_ylim(BP_MIN_YLIMS[lam_idx], top=BP_MAX_YLIMS[lam_idx])
    ax.set_xlabel("k", fontsize=20)
    ax.set_ylabel("Value", fontsize=20)
    ax.set_title(f"{latex_symbol}", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    if lam_idx == 2:
        ax.legend(fontsize=20, loc=1, framealpha=0.5)

# Optional: nicer spacing + global title
# fig.suptitle(f"Bootstrap PF band plots (quantiles); J={J}; N={N}; "
#              + f"Δt={delta_t}; l={l}", fontsize=24)
plt.tight_layout()

if SAVE_PLOTS:
    folder_path = Path(FOLDER_PATH_STR)
    folder_path.mkdir(parents=True, exist_ok=True)

    image_name = "all_bandplots_quantiles.png"
    plt.savefig(folder_path / image_name, bbox_inches='tight')
    plt.close(fig)
else:
    plt.show()

# %%

## ESS ##

plt.figure(figsize=(8, 3))
plt.plot(k_series, pf_boot.summaries.ESSs, color="red")
plt.xlabel("k", fontsize=16)
plt.ylabel("ESS", fontsize=16)
plt.tick_params(axis='both', labelsize=16)
# plt.title("ESS over time: Bootstrap PF\n"
#           + f"J={J}; N={N}; $\Delta t$={delta_t}; l={l}",
#           fontsize=20)
if SAVE_PLOTS:
    folder_path = Path(FOLDER_PATH_STR)
    folder_path.mkdir(parents=True, exist_ok=True)
    image_name = "ess.png"
    plt.savefig(folder_path / image_name,
                bbox_inches='tight')
    plt.close("all")
else:
    plt.show()

# %%

## Choose some k between 0 and K+1 (inclusive) ##

k = np.random.randint(K+1)

# %%

## KDE for each lam (uses weights of particles) ##

if not SAVE_PLOTS and PLOT_EXTRAS:
    for lam in true_lams.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(x=(ds_boot["Exp_X"].sel({'lam': lam, 'k': k})
                       .values.reshape(-1)),
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

if not SAVE_PLOTS and PLOT_EXTRAS:
    plot_df = (
        ds_boot["Exp_X"].sel(k=k)
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

## Band plots using summaries.moments (plots log-rates) ##

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
