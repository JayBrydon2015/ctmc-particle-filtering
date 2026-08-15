# -*- coding: utf-8 -*-

""" Simulation of the transition rates using the CTMC state-space models. """


#### IMPORTS ####

from pathlib import Path
import numpy as np
from ctmc_modules.ctmc_ssms import (
    CTMC,
    GP_CTMC,
    lams_idx_to_gen_pos
)
from testing.run_results_generation import (
    TEXTWIDTH_IN,
    FONT_SIZE,
    TICK_FONT_SIZE,
    get_latex_rate_symbol,
    apply_font_sizes
)


#### CONSTANTS ####

SAVE_PLOTS = True

PLOTS_ROOT_FOLDER_DIR = Path(__file__).parent / "generated_plots"
PLOTS_ROOT_FOLDER_DIR.mkdir(exist_ok=True)

PLOTS_FOLDER_DIR = PLOTS_ROOT_FOLDER_DIR / "True_rate_simulations"
PLOTS_FOLDER_DIR.mkdir(exist_ok=True)


#### MATPLOTLIB IMPORTS ####

if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg') # Must be called before importing pyplot
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


#### CTMC SSM CONSTANTS ####

n = 2
J = 1
delta_t = 0.03
K = 100
t_K = 3
assert delta_t * K == t_K

k_series = np.arange(K + 1)
time_series = delta_t * k_series

mu0  = np.array([10,   10  ])
var0 = np.array([0.25, 0.25])

num_simulations = 5


#### FUNCTIONS ####


# %%

#### SIMULATE GAMMA SSM; PLOT RATE TRAJECTORIES ####

num_params = 4
params_dict = {
    "tv_type": ["PV", "PV", "CV", "CV"],
    "C": [1, 10, 1, 10],
}
assert all(len(value) == num_params for value in params_dict.values())

fig, axes = plt.subplots(
    2, 2,
    figsize=(0.7*TEXTWIDTH_IN, 2.5),
    sharex=True,
    sharey=True
)
axes = axes.flatten()

for i in range(num_params):

    tv_type = params_dict["tv_type"][i]
    C = params_dict["C"][i]

    ax = axes[i]

    ctmc_ssm = CTMC(
        n=n,
        J=J,
        delta_t=delta_t,
        tv_type=tv_type,
        C=C,
        mu0=mu0,
        var0=var0
    )

    for j in range(num_simulations):

        true_rates, _ = ctmc_ssm.simulate(T=K+1)

        # Shape becomes (K+1, 2)
        true_rates = np.vstack(true_rates)
        
        p, q = lams_idx_to_gen_pos(0, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        # L_1_2
        ax.plot(
            time_series,
            true_rates[:, 0],
            color="tab:blue",
            alpha=0.7,
            lw=1,
        )
    
    if tv_type == "CV":
        tv_type_title = "CV"
    elif tv_type == "PV":
        tv_type_title = "PV"
    else:
        raise NotImplementedError
    
    ax.set_title(rf"$\mathrm{{{tv_type_title}}},\ C={C}$", pad=1)
    ax.grid(True)

for ax in axes[2:]:
    ax.set_xlabel("time")

for ax in axes[::2]:
    ax.set_ylabel(latex_symbol, labelpad=2.1)

plt.subplots_adjust(
    hspace=0.3,
    wspace=0.08,
)

apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)

if SAVE_PLOTS:
    fig.savefig(
        PLOTS_FOLDER_DIR / "true_rate_simulations.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
else:
    plt.show()


# %%

#### SIMULATE GAMMA SSM; PLOT RATE TRAJECTORIES ####

num_params = 4
params_dict = {
    "kappa":  [0.2,  0.95, 0.2,  0.95],
    "sigma2": [0.1,  0.1,  0.3,  0.3 ],
}
assert all(len(value) == num_params for value in params_dict.values())

fig, axes = plt.subplots(
    2, 2,
    figsize=(0.7*TEXTWIDTH_IN, 2.5),
    sharex=True,
    sharey=True
)
axes = axes.flatten()

for i in range(num_params):

    kappa = params_dict["kappa"][i]
    sigma2 = params_dict["sigma2"][i]

    ax = axes[i]

    ctmc_ssm = GP_CTMC(
        n=n,
        J=J,
        delta_t=delta_t,
        kappa=kappa,
        sigma2=sigma2,
        mu0=mu0
    )

    for j in range(num_simulations):
        
        true_log_rates, _ = ctmc_ssm.simulate(T=K+1)
        
        # Shape becomes (K+1, 2)
        true_log_rates = np.vstack(true_log_rates)
        
        # Convert log-rates to rates
        true_rates = np.exp(true_log_rates)
        
        p, q = lams_idx_to_gen_pos(0, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        # L_1_2
        ax.plot(
            time_series,
            true_rates[:, 0],
            color="tab:blue",
            alpha=0.7,
            lw=1,
        )
    
    ax.set_title(rf"$\kappa={kappa},\, \sigma^2={sigma2}$", pad=3)
    ax.grid(True)

for ax in axes[2:]:
    ax.set_xlabel("time")

for ax in axes[::2]:
    ax.set_ylabel(latex_symbol, labelpad=2.1)

plt.subplots_adjust(
    hspace=0.3,
    wspace=0.08,
)

apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)

if SAVE_PLOTS:
    fig.savefig(
        PLOTS_FOLDER_DIR / "true_rate_gp_simulations.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
else:
    plt.show()

