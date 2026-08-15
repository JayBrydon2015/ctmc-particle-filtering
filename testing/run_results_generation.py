# -*- coding: utf-8 -*-

"""

Functionality that generates the results of a run for any of the CTMC case
study examples.

"""


###### IMPORTS ######

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments

from ctmc_modules.ctmc_ssms import (
    lams_idx_to_gen_pos,
    CTMC,
    GP_CTMC,
    get_default_rw_initial_config,
    get_rw_all_state_p_initial_config
)

from rates_simulation.true_rates_simulation_funtions import (
    simulate_example_a,
    simulate_sine_rates_n2,
    simulate_example_c,
    simulate_case_study_4,
    simulate_data,
    simulate_data_manually_example_a
)


###### CONSTANTS ######

SAVE_PLOTS  = True  # Save plots to a folder (True) or show them (False)
PLOT_EXTRAS = False # Plot extra stuff (True rates, KDEs and PW scatter plots)

PLOTS_ROOT_FOLDER_DIR = Path(__file__).parent / "generated_plots"
PLOTS_ROOT_FOLDER_DIR.mkdir(exist_ok=True)

CASE_STUDY_NUMS = [1, 2, 3, 4]

RUN_LABEL_Y_OFFSETS_2_RUNS = {
    0: 0.07,
    1: 0.00,
}

RUN_LABEL_Y_OFFSETS_3_RUNS = {
    0: 0.08,
    1: 0.04,
    2: -0.01,
}


###### PLOTTING LIBRARY IMPORTS & PLOTTING CONSTANTS ######

if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg') # Must be called before importing pyplot
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

TEXTWIDTH_IN = 6.614
LATEX_FONT_SIZE = 12
FONT_SIZE = LATEX_FONT_SIZE - 1
TICK_FONT_SIZE = FONT_SIZE - 1

ESS_FORMATTER = ScalarFormatter(useMathText=False)
ESS_FORMATTER.set_scientific(True)
ESS_FORMATTER.set_powerlimits((4, 4))


###### HELPER FUNCTIONS ######


def get_p_q_from_lam_col_name(lam_col: str):
    """Given L_p_q, returns p and q as integers."""
    p, q = lam_col.split("_")[1:]
    return int(p), int(q)


def get_latex_rate_symbol_cs1(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate used in the image file
        name. For Example A only.
    """
    
    if p == 1 and q == 2:
        return "$\\varepsilon$", "Eps"
    elif p == 2 and q == 3:
        return "$\\delta$", "Delta"
    elif p == 3 and q == 1:
        return "$\\phi$", "Phi"
    else:
        return f"$\\lambda^{{{p} \\to {q}}}$", f"L_{p}_{q}"


def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate that could be used in
        the image file's name, for example.
    """
    return f"$\\lambda^{{{p} \\to {q}}}$", f"L_{p}_{q}"


def weighted_quantile(values, weights, quantiles):
    """
    Calculates weighted quantiles using linear interpolation.

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


def map_num_to_letter(num: int) -> str:
    """Maps 0 to 'A', 1 to 'B', 2 to 'C', up to 25 to 'Z'."""
    if 0 <= num <= 25:
        return chr(65 + num)
    raise ValueError("Input must be between 0 and 25 inclusive.")


def apply_font_sizes(fig, font_size, tick_font_size):
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


def make_true_rates_dataframe(*, true_rates, n):
    """
    Creates a pandas data frame storing the true transition rates. The columns
    of the data frame refer to each of the transition rates and each column's
    name is 'L_p_q', referring to the transition rate from states p to q.
    
    Assumes the rates are ordered as L_1_2, L_1_3, ..., L_1_n, L_2_1,
    L_2_3, ..., L_2_n, ..., L_n_1, L_n_2, ..., L_n_n-1, where L_p_q refers
    to the transition rate from states p to q. Note that p and q are between
    1 and n, and p != q always.
    
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
    
    
    ###### Run the bootstrap particle filter ######
    
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
    
    
    ###### Store lambda particles and weights in an xarray.Dataset ######
    
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
    
    
    ###### Calculate quantiles and add into ds_boot ######
    
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


def generate_run_results(*,
        n: int,
        delta_t: float,
        K: int,
        J: int,
        ssm_params: dict,
        case_study_num: int,
        gp: bool,
        N: int = None,
        skip_particle_filter: bool = False,
    ):
    
    """
    Generate the results of a run in a group for one of the examples.
    Return the results as a dictionary.
    
    Inputs
    ------
    ctmc_params: dictionary of CTMC SSM parameters, specifically those that
        differ between the Gamma transitions SSM and the Gaussian process SSM.
    example: one of "a", "b", "c", "d", depending on which example.
    gp: True if modelling log-rates in the hidden process; False otherwise.
    skip_particle_filter: True if skipping the particle filter.
    """
    
    if not skip_particle_filter and N is None:
        raise ValueError("Please pass in the number of particles N!")
    
    if case_study_num not in CASE_STUDY_NUMS:
        raise ValueError(
            f"case_study_num must be one of {CASE_STUDY_NUMS}. "
            f"Currently, it is {case_study_num}."
        )
    
    k_series    = np.arange(K + 1)   # [0, 1, ..., K]
    time_series = delta_t * k_series # [t_0, t_1, ..., t_K], given t_0 = 0
    
    exp_X = True if gp else False
    
    ## Unpack the CTMC parameters ##
    
    if not gp:
        mu0 = ssm_params["mu0"]
        var0 = ssm_params["var0"]
        tv_type = ssm_params["TV"]
        C = ssm_params["C"]
    else:
        mu = ssm_params["mu"]
        kappa = ssm_params["kappa"]
        sigma2 = ssm_params["sigma2"]
    
    ## Define the parameters not altered between runs ##
    ## but unique to the example ##
    
    if case_study_num == 1:
        y_init = get_rw_all_state_p_initial_config(J=J)
        true_rates = simulate_example_a(K=K) # For data simulation
        
    elif case_study_num == 2:
        y_init = None
        true_rates = simulate_sine_rates_n2(K=K) # For data simulation
        
    elif case_study_num == 3:
        y_init = None
        true_rates = simulate_example_c(K=K) # For data simulation
    
    elif case_study_num == 4:
        y_init = get_rw_all_state_p_initial_config(J=J)
        true_rates = simulate_case_study_4(K=K) # For data simulation
        
    else:
        raise NotImplementedError(
            f"Case Study #{case_study_num} has not been implemented."
        )
    
    ## Create CTMC SSM ##
    
    if not gp:
    
        ctmc_ssm = CTMC(
            n = n,
            J = J,
            delta_t = delta_t,
            tv_type = tv_type,
            C = C,
            mu0 = mu0,
            var0 = var0,
            y_init = y_init,
            px_verbose = True
        )
    
    else:
    
        ctmc_ssm = GP_CTMC(
            n = n,
            J = J,
            delta_t = delta_t,
            kappa = kappa,
            sigma2 = sigma2,
            mu = mu,
            y_init = y_init,
            px_verbose = True
        )
    
    if y_init is None:
        y_init = ctmc_ssm.y_init
    
    ## Simulate data ##
    
    if case_study_num == 1 and J == 1 and K == 300:
        
        # Manual simulation of the data only occurs in example A
        data = simulate_data_manually_example_a()
        
    else:
        
        data = simulate_data(
            true_rates=true_rates,
            n=n,
            J=J,
            delta_t=delta_t,
            y_init=y_init
        )
    
    ## Run particle filter ##
    
    ds_boot = None
    
    if not skip_particle_filter:
    
        ds_boot = run_particle_filter(
            ctmc_ssm = ctmc_ssm,
            data = data,
            N = N,
            k_series = k_series,
            exp_X = exp_X,
        )
    
    return {
        "k_series": k_series,
        "time_series": time_series,
        "data": data,
        "ds_boot": ds_boot
    }


###### PLOT GENERATION FUNCTIONS #######


#### STACK ALL PLOTS VERTICALLY (SPECIAL CASE) ####


def stack_all_plots_vertically(*,
        n: int,
        time_series,
        true_rates_df_plot,
        data,
        ds_boot,
        plots_folder_dir: Path,
    ):
    
    """ Stack all plots vertically. Special case when case study #1 and
        J == 1.
    """
    
    K_plot = true_rates_df_plot.shape[0] - 1
    
    t_K = time_series[-1]
    time_series_true_rates = np.linspace(0, t_K, K_plot + 1)
    
    ## Plot parameters ##
    
    bp_yticks = [
        [0, 2, 4, 6],
        [0, 2, 4],
        [0, 2, 4, 6],
        [0, 5, 10],
    ]
    
    ## Initialise subplots ##
    
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 8.8))

    outer_gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.6, 1.0 + 1.7 * 4],
        hspace=0.07   # spacing between data plot and ESS plot block
    )
    
    rw_ax = fig.add_subplot(outer_gs[0])
    
    rest_gs = outer_gs[1].subgridspec(
        nrows=5,
        ncols=1,
        height_ratios=[1.0] + [1.7] * 4,
        hspace=0.12   # spacing between ESS and band plots
    )
    
    ess_ax = fig.add_subplot(rest_gs[0], sharex=rw_ax)
    
    band_axes = [
        fig.add_subplot(rest_gs[i], sharex=rw_ax)
        for i in range(1, 5)
    ]
    
    axes = [rw_ax, ess_ax] + band_axes
    
    ## Data plot ##
    
    data_plot = np.vstack(data) + 1
    
    rw_ax.plot(time_series, data_plot[:, 0], lw=2)
    
    rw_ax.set_ylim(0.8, 3.2)
    rw_ax.set_yticks([1, 2, 3])
    rw_ax.set_ylabel("State")
    rw_ax.grid(True)
    
    ## ESS plot ##
    
    ess_ax.plot(
        time_series,
        ds_boot['ESS'],
        color="red",
        lw=2
    )
    
    ess_ax.set_ylabel("ESS")
    ess_ax.ticklabel_format(
        axis='y',
        style='sci',
        scilimits=(0, 0)
    )
    ess_ax.yaxis.set_major_formatter(ESS_FORMATTER)
    ess_ax.grid(True)
    
    ## Band plots ##
    
    band_ax_idx = 0
    
    for lam_idx, lam in enumerate(true_rates_df_plot.columns):
        
        if lam_idx in {1, 5}:
            continue
        
        ax = band_axes[band_ax_idx]
        
        p, q = lams_idx_to_gen_pos(lam_idx, n)
        
        latex_symbol, _ = get_latex_rate_symbol_cs1(p, q)
        
        median = ds_boot["X_quantiles"].sel(
            lam=lam,
            quantile=0.5
        )
        
        lq = ds_boot["X_quantiles"].sel(
            lam=lam,
            quantile=0.05
        )
        
        uq = ds_boot["X_quantiles"].sel(
            lam=lam,
            quantile=0.95
        )
        
        ax.plot(
            time_series_true_rates,
            true_rates_df_plot[lam].values,
            color="blue",
            lw=2
        )
        
        ax.plot(
            time_series,
            median,
            color="orange",
            lw=2
        )
        
        ax.fill_between(
            time_series,
            lq,
            uq,
            color="orange",
            alpha=0.3
        )
        
        ax.set_yticks(bp_yticks[band_ax_idx])
        ax.set_ylabel(latex_symbol)
        ax.grid(True)
        
        band_ax_idx += 1
        
    ## Formatting options ##
    
    X_LIM = (-0.1, 3.1)

    for ax in axes:
        ax.set_xlim(X_LIM)
        
    axes[-1].set_xlabel("time")

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    
    if SAVE_PLOTS:
        fig.savefig(
            plots_folder_dir / "J1_all_plots_combined.pdf",
            format="pdf",
            bbox_inches='tight',
        )
        plt.close(fig)
    else:
        plt.show()


#### BAND PLOTS & OPTIONAL STATE PROPORTIONS PLOT COMBINED ####


def band_plots_and_optional_state_proportions(*,
        case_study_num: int,
        num_runs: int,
        run_letters: list[str],
        n: int,
        true_rates_df_plot, # pandas DF
        group_results: dict,
        band_nrows: int,
        band_ncols: int,
        plot_state_proportions_bool: list[bool],
        bp_ymin: list,
        bp_ymax: list,
        y_ticks: list,
        plots_folder_dir: Path,
    ):
    
    """ Band plots and optional state proportions plots stacked vertically
        for each run in a group.
    """
    
    if case_study_num not in CASE_STUDY_NUMS:
        raise ValueError(
            f"case_study_num must be one of {CASE_STUDY_NUMS}. "
            f"Currently, it is {case_study_num}."
        )
    
    K_plot = true_rates_df_plot.shape[0] - 1
    
    
    # ============================================================
    # Configuration
    # ============================================================
    
    # Spacing BETWEEN rows of band plots
    band_hspace = 0.15
    
    # Spacing BETWEEN columns of band plots
    if band_nrows == 1:
        band_wspace = 0.23
    elif band_nrows == 2:
        band_wspace = 0.32
    else:
        raise NotImplementedError
    
    # Spacing BETWEEN band plots and the PBC plot
    pbc_hspace = 0.24
    
    # Spacing BETWEEN separate run blocks
    if band_nrows == 1:
        
        outer_hspace = 0.3
        
    elif band_nrows == 2:
        
        if num_runs == 2:
            outer_hspace = 0.16 if plot_state_proportions_bool[-1] else 0.19
        elif num_runs == 3:
            outer_hspace = 0.20
        else:
            outer_hspace = 0.20
    
    # X, Y label padding
    y_lab_pad = 1.5
    
    
    # Give runs with PBC a bit more height
    outer_height_ratios = [
        1.6 if plot_state_proportions_bool[i] else 1.0
        for i in range(num_runs)
    ]
    
    if band_nrows == 1:
        per_block_height = 1.5
    elif band_nrows == 2:
        per_block_height = 2.5
    else:
        raise NotImplementedError
    
    # Figure height -- adjust as necessary
    fig = plt.figure(
        figsize=(
            TEXTWIDTH_IN,
            per_block_height * sum(outer_height_ratios)
        )
    )
    
    outer_gs = fig.add_gridspec(
        nrows=num_runs,
        ncols=1,
        height_ratios=outer_height_ratios,
        hspace=outer_hspace
    )
    
    
    # Store axes for later run-label placement
    run_block_axes = []
    
    
    # ============================================================
    # Loop over runs
    # ============================================================
    
    for run_idx in range(num_runs):
    
        run_letter = run_letters[run_idx]
        plot_pbc = plot_state_proportions_bool[run_idx]
    
        time_series = group_results[run_letter]["time_series"]
        data = group_results[run_letter]["data"]
        ds_boot = group_results[run_letter]["ds_boot"]
    
        t_K = time_series[-1]
        time_series_true_rates = np.linspace(0, t_K, K_plot + 1)
    
    
        # ========================================================
        # Inner layout for this run
        # ========================================================
    
        if plot_pbc:
    
            # --------------------------------------------
            # Top = band plots
            # Bottom = state proportions
            # --------------------------------------------
            block_gs = outer_gs[run_idx].subgridspec(
                nrows=2,
                ncols=1,
                height_ratios=[1.0, 0.6],
                hspace=pbc_hspace
            )
    
            band_gs = block_gs[0].subgridspec(
                nrows=band_nrows,
                ncols=band_ncols,
                hspace=band_hspace,
                wspace=band_wspace
            )
    
        else:
    
            # --------------------------------------------
            # Band plots only
            # --------------------------------------------
            block_gs = outer_gs[run_idx].subgridspec(
                nrows=1,
                ncols=1
            )
    
            band_gs = block_gs[0].subgridspec(
                nrows=band_nrows,
                ncols=band_ncols,
                hspace=band_hspace,
                wspace=band_wspace
            )
    
    
        # ========================================================
        # Create band-plot axes
        # ========================================================
    
        band_axes = np.empty(
            (band_nrows, band_ncols),
            dtype=object
        )
    
        for row in range(band_nrows):
            for col in range(band_ncols):
    
                # Share x-axis with the first band plot
                if row == 0 and col == 0:
                    ax = fig.add_subplot(band_gs[row, col])
                else:
                    ax = fig.add_subplot(
                        band_gs[row, col],
                        sharex=band_axes[0, 0]
                    )
    
                band_axes[row, col] = ax
    
    
        flat_band_axes = band_axes.flatten()
    
    
        # ========================================================
        # Plot band plots
        # ========================================================
    
        for lam_idx, lam in enumerate(true_rates_df_plot.columns):
            
            ax = flat_band_axes[lam_idx]
            
            # p, q = lams_idx_to_gen_pos(lam_idx, n)
            p, q = get_p_q_from_lam_col_name(lam)
            
            if case_study_num == 1:
                latex_symbol, _ = get_latex_rate_symbol_cs1(p, q)
            else:
                latex_symbol, _ = get_latex_rate_symbol(p, q)
            
            median = ds_boot["X_quantiles"].sel(
                lam=lam,
                quantile=0.5
            )
            
            lq = ds_boot["X_quantiles"].sel(
                lam=lam,
                quantile=0.05
            )
            
            uq = ds_boot["X_quantiles"].sel(
                lam=lam,
                quantile=0.95
            )
            
            ax.plot(
                time_series_true_rates,
                true_rates_df_plot[lam].values,
                label="True rate",
                color="blue",
                alpha=0.8,
                lw=1
            )
    
            ax.plot(
                time_series,
                median,
                label="PF median",
                color="orange",
                alpha=0.7,
                lw=1
            )
    
            ax.fill_between(
                time_series,
                y1=lq,
                y2=uq,
                color="orange",
                alpha=0.3,
                lw=1
            )
    
            ax.set_ylabel(
                latex_symbol,
                labelpad=y_lab_pad
            )
    
            ax.set_xlim(0, t_K)
    
            ax.set_xticks([
                0,
                t_K / 3,
                2 * t_K / 3,
                t_K
            ])
    
            if bp_ymin is not None and bp_ymax is not None:
                
                ylim_lower_offset = -0.2 if bp_ymin[lam_idx] == 0 else 0
                ylim_upper_offset = (
                    0.5 if bp_ymax[lam_idx] == y_ticks[lam_idx][-1] else 0
                )
                
                ax.set_ylim(
                    bottom=bp_ymin[lam_idx] + ylim_lower_offset,
                    top=bp_ymax[lam_idx] + ylim_upper_offset
                )
    
            if y_ticks is not None:
                ax.set_yticks(y_ticks[lam_idx])
    
    
        # ========================================================
        # X-axis formatting for band plots
        # ========================================================
    
        # Hide x tick labels everywhere initially
        for ax in flat_band_axes:
            ax.tick_params(
                axis="x",
                labelbottom=False
            )
            ax.set_xlabel("")
    
    
        # If there is a PBC plot, only the bottom row of the
        # band plots gets x tick labels.
        #
        # If there is no PBC plot, the bottom row of the band
        # plots is the bottom of the entire block.
        bottom_band_axes = band_axes[-1, :]
    
        for ax in bottom_band_axes:
            ax.tick_params(
                axis="x",
                labelbottom=True
            )
    
    
        # ========================================================
        # State proportions plot
        # ========================================================
    
        pbc_ax = None
    
        if plot_pbc:
    
            pbc_ax = fig.add_subplot(
                block_gs[1],
                sharex=band_axes[0, 0]
            )
    
            counts = np.array([
                np.bincount(
                    data_k.reshape(-1),
                    minlength=n
                )
                for data_k in data
            ])
    
            proportions = (
                counts /
                np.array([
                    len(data_k.reshape(-1))
                    for data_k in data
                ])[:, None]
            )
    
            labels = [
                f"{value + 1}"
                for value in range(n)
            ]
    
            pbc_ax.stackplot(
                time_series,
                proportions.T,
                labels=labels,
                alpha=0.7,
                edgecolor="white",
                linewidth=1
            )
    
            pbc_ax.set_ylabel("Proportion")
    
            pbc_ax.set_xlim(0, t_K)
    
            pbc_ax.set_xticks([
                0,
                t_K / 3,
                2 * t_K / 3,
                t_K
            ])
    
            pbc_ax.set_ylim(0, 1)
            pbc_ax.set_yticks([
                0.0,
                0.5,
                1.0
            ])
    
            # Show state legend only once
            if run_idx == num_runs - 1:
    
                pbc_ax.legend(
                    ncols=n,
                    title="State",
                    loc="lower right",
                    framealpha=0.5
                )
    
    
        # ========================================================
        # Store axes for run-label placement
        # ========================================================
    
        run_block_axes.append({
            "run_letter": run_letter,
            "band_axes": band_axes,
            "pbc_ax": pbc_ax
        })
    
    
    # ============================================================
    # Global formatting
    # ============================================================
    
    apply_font_sizes(
        fig,
        FONT_SIZE,
        TICK_FONT_SIZE
    )
    
    
    # ============================================================
    # Add run labels on the left and x label 'time' at bottom
    # ============================================================
    
    if band_nrows == 1:
        RUN_LABEL_X = 0.038
    elif band_nrows == 2:
        RUN_LABEL_X = 0.045
    else:
        raise NotImplementedError
    
    for run_idx, block_info in enumerate(run_block_axes):
    
        run_letter = block_info["run_letter"]
        band_axes = block_info["band_axes"]
        pbc_ax = block_info["pbc_ax"]
    
        top_left_ax = band_axes[0, 0]
    
        if pbc_ax is None:
    
            # Centre over the band-plot block
            pos = top_left_ax.get_position()
    
            # Need the bottom-left band axis as well
            bottom_left_ax = band_axes[-1, 0]
            pos_bottom = bottom_left_ax.get_position()
    
            y_center = 0.5 * (
                pos.y1 + pos_bottom.y0
            )
    
        else:
    
            # Centre over the entire run block:
            # band plots + PBC
            pos_top = top_left_ax.get_position()
            pos_bottom = pbc_ax.get_position()
    
            y_center = 0.5 * (
                pos_top.y1 + pos_bottom.y0
            )
    
        fig.text(
            RUN_LABEL_X,
            y_center,
            f"Run {run_letter}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=FONT_SIZE
        )
    
    if plot_state_proportions_bool[-1]:
        
        fig.supxlabel("time", y=0.04)
        
    else:
        
        last_block = run_block_axes[-1]
        last_band_axes = last_block["band_axes"]
        
        for ax in last_band_axes[-1, :]:
            ax.set_xlabel("time")
    
    
    # ============================================================
    # Save/show
    # ============================================================
    
    if SAVE_PLOTS:
    
        fig.savefig(
            plots_folder_dir /
            "group_bandplots_and_optional_pbc.pdf",
            format="pdf",
            bbox_inches="tight"
        )
    
        plt.close(fig)
    
    else:
    
        plt.show()


#### 2x2 BAND PLOTS FOR CASE STUDY #3 ####


def band_plots_cs3(*,
        num_runs: int,
        run_letters: list[str],
        group_results: dict,
        true_rates_df_plot,
        n: int,
        bp_ymin: list,
        bp_ymax: list,
        y_ticks: list,
        plots_folder_dir: Path,
    ):
    
    """ 1x2 or 2x2 band plots for one transition rate, each corresponding to
        one of two or four runs, respectively. Used for Case Study #3.
    """
    
    if num_runs != 2 and num_runs != 4:
        raise ValueError(
            f"This plotting code expects 2 or 4 runs, but got {num_runs}."
        )
    
    assert len(run_letters) == num_runs
    
    lam_idx = 0 # the specific rate to plot: L_1_2
    
    K_plot = true_rates_df_plot.shape[0] - 1
    
    n_rows = 1 if num_runs == 2 else 2
    
    fig_height = 2.0 if num_runs == 2 else 3.6
    
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(TEXTWIDTH_IN, fig_height),
        sharex=True
    )
    
    axes = axes.flatten()
    
    # Code below this should be modified
    
    for run_idx in range(num_runs):
        
        ax = axes[run_idx]
        
        run_letter = run_letters[run_idx]
    
        time_series = group_results[run_letter]["time_series"]
        ds_boot = group_results[run_letter]["ds_boot"]
    
        t_K = time_series[-1]
        time_series_true_rates = np.linspace(0, t_K, K_plot + 1)
    
        lam = true_rates_df_plot.columns[lam_idx]
    
        p, q = lams_idx_to_gen_pos(lam_idx, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
    
        median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
        lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
        uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    
        ax.plot(
            time_series_true_rates,
            true_rates_df_plot[lam].values,
            label="True rate",
            color="blue",
            alpha=0.8,
            lw=1
        )
    
        ax.plot(
            time_series,
            median,
            label="PF median",
            color="orange",
            alpha=0.7,
            lw=1
        )
    
        ax.fill_between(
            time_series,
            y1=lq,
            y2=uq,
            color="orange",
            alpha=0.3,
            lw=1
        )
    
        ax.set_title(f"Run {run_letter}", pad=3)
        ax.set_ylabel(latex_symbol, labelpad=1.5)
    
        ax.set_xlim(0, t_K)
        ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
        
        if bp_ymin is not None and bp_ymax is not None:
            ax.set_ylim(
                bottom=bp_ymin[lam_idx], # - 0.1,
                top=bp_ymax[lam_idx] # + 0.1
            )
        
        if y_ticks is not None:
            ax.set_yticks(y_ticks[lam_idx])
    
    # Hide x tick labels on top row
    for ax in axes[:-2]:
        ax.tick_params(axis="x", labelbottom=False)
    
    # Bottom row gets x-axis labels
    for ax in axes[-2:]:
        ax.set_xlabel("time")
    
    # Only show legend once
    # axes[1].legend(loc="upper right", framealpha=0.5)
    
    plt.subplots_adjust(
        hspace=0.26,
        wspace=0.25
    )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    
    if SAVE_PLOTS:
        fig.savefig(
            plots_folder_dir / "single_rate_runs_bandplots.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


#### STANDALONE STATE PROPORTIONS PLOT ####


def state_proportions_standalone_plot(*,
        example_folder_name: str = None,
        case_study_num: int = None,
        delta_t: float,
        K: int = None,
        J: int = 10000,
        t_K: float = None,
        n: int = None,
        data: list = None,
    ):
    
    """ Standalone state proportions plot. Pass in case study number, in which
        the data of the case study will be simulated and plotted, or pass in
        data to plot.
    """
    
    if case_study_num not in CASE_STUDY_NUMS:
        raise ValueError(
            f"case_study_num must be one of {CASE_STUDY_NUMS}. "
            f"Currently, it is {case_study_num}."
        )
    
    if example_folder_name is None:
        example_folder_name = f"CTMC_CaseStudy{case_study_num}_Figs"
    
    PLOTS_FOLDER_DIR = PLOTS_ROOT_FOLDER_DIR / f"{example_folder_name}"
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    
    #### Generate data ####
    
    if data is None:
    
        assert delta_t is not None
    
        if t_K is None:
            t_K = K * delta_t
        
        assert K * delta_t == t_K
        
        if case_study_num == 1:
            n = 3
            y_init = get_rw_all_state_p_initial_config(J=J, p=1, n=n)
            true_rates = simulate_example_a(K=K)
            
        elif case_study_num == 2:
            n = 2
            y_init = get_default_rw_initial_config(n=n, J=J)
            true_rates = simulate_sine_rates_n2(K=K)
            
        elif case_study_num == 3:
            n = 2
            y_init = get_default_rw_initial_config(n=n, J=J)
            true_rates = simulate_example_c(K=K)
            
        elif case_study_num == 4:
            n = 5
            y_init = get_rw_all_state_p_initial_config(J=J, p=1, n=n)
            true_rates = simulate_case_study_4(K=K)
        
        data = simulate_data(
            true_rates = true_rates,
            n = n,
            J = J,
            delta_t = delta_t,
            y_init = y_init
        )
        
    else:
        
        assert n is not None
        
        if K is None:
            K = len(data) - 1
        
        if t_K is None:
            t_K = K * delta_t
    
    k_series = np.arange(K+1)
    time_series = delta_t * k_series
    
    
    #### GENERATE STANDALONE STATE PROPORTIONS PLOT ####
    
    counts = np.array([
        np.bincount(data_k.reshape(-1), minlength=n)
        for data_k in data
    ])
    
    proportions = (
        counts /
        np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
    )
    
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.2))
    
    labels = [f"{value + 1}" for value in range(n)]
    
    ax.stackplot(
        time_series,
        proportions.T,
        labels=labels,
        alpha=0.7,
        edgecolor="white",
        linewidth=1
    )
    
    ax.set_xlabel("time")
    ax.set_ylabel("Proportion")
    
    ax.set_xlim(0, t_K)
    ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    
    ax.legend(
        ncols=n,
        title="State",
        loc="lower left",
        framealpha=0.5
    )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    
    fig.tight_layout()
    
    if SAVE_PLOTS:
        fig.savefig(
            PLOTS_FOLDER_DIR / "state_proportions.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


#### DATA & ESS PLOT (SPECIAL CASE, CASE STUDY #1) ####


def plot_data_ess_special_case_cs1(*,
        time_series,
        data,
        ess,
        plots_folder_dir: Path,
    ):
    
    """ Plot data and ESS side by side. Only for case study #1 and
        when J = 8.
    """
    
    t_K = time_series[-1]
    
    data_plot = np.vstack(data)
    data_plot += 1  # '0' -> State 1, etc
    
    n_walkers = data_plot.shape[1]
    
    if n_walkers != 8:
        raise ValueError(
            f"Expected 8 random walkers for this run, "
            f"but data_plot has shape {data_plot.shape}."
        )
    
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.6))

    outer_gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[2, 1.15],
        wspace=0.18
    )
    
    data_gs = outer_gs[0].subgridspec(
        nrows=4,
        ncols=2,
        hspace=0.25,
        wspace=0.20
    )
    
    axes = []
    for row in range(4):
        for col in range(2):
            share_ax = axes[0] if axes else None
            axes.append(
                fig.add_subplot(data_gs[row, col], sharex=share_ax)
            )
    
    ess_ax = fig.add_subplot(outer_gs[1])
    
    for j in range(n_walkers):
        axes[j].plot(time_series, data_plot[:, j], lw=2)
        axes[j].grid(True)
        axes[j].set_ylim(0.8, 3.2)
        axes[j].set_xlim(0, t_K)
    
    for ax in axes[:-2]:
        ax.tick_params(axis="x", labelbottom=False)
    
    for ax in axes[-2:]:
        ax.set_xlabel("time")
        
    for ax in axes:
        ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
    
    ess_ax.plot(time_series, ess, color="red", lw=1)
    ess_ax.set_xlabel("time")
    ess_ax.set_xlim(0, t_K)
    ess_ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
    ess_ax.set_ylabel("ESS")
    ess_ax.yaxis.set_major_formatter(ESS_FORMATTER)
    ess_ax.grid(True)
    
    pos = ess_ax.get_position()

    height_frac = 0.65
    new_height = pos.height * height_frac
    new_y0 = pos.y0 + 0.5 * (pos.height - new_height)
    
    ess_ax.set_position([
        pos.x0 + 0.05,
        new_y0,
        pos.width * 0.85,
        new_height
    ])
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    
    if SAVE_PLOTS:
        fig.savefig(
            plots_folder_dir / "J8_data_ess.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


#### TRUE TRANSITIONS RATE PLOT: CASE STUDY #2 ####


def generate_true_rates_plot_example_b(*,
        example_folder_name: str,
        n: int,
        true_lams_df,
        t_K: float = 3
    ):
    
    
    """ Creates and saves the plot of the true transition rates for
        the second case study on sinusoidal transition rates.
    """
    
    TRUE_LAMS_PLOT_DIR = (
        PLOTS_ROOT_FOLDER_DIR /
        f"{example_folder_name}"
    )
    
    K_plot = true_lams_df.shape[0] - 1
    time_series_plot = np.linspace(0, t_K, K_plot + 1)
    
    fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH_IN, 1.6))
    
    lines = []
    
    for lam_idx, lam in enumerate(true_lams_df.columns):
        
        p, q = lams_idx_to_gen_pos(lam_idx, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        line, = ax.plot(
            time_series_plot,
            true_lams_df[lam],
            lw=2
        )
        
        lines.append((line, latex_symbol, lam_idx))
    
    ax.set_xlim(-0.05, 3.55) # more space on right so labels fit
    ax.set_xlabel("time")
    ax.set_ylim(-0.1, 4.6)
    ax.set_yticks(np.arange(0, 5))
    
    # Label each curve near its right-hand end
    x_text = time_series_plot[-1] + 0.06
    y_offsets = [-0.16, -0.16] # small offsets to reduce overlap
    
    for (line, latex_symbol, lam_idx), y_offset in zip(lines, y_offsets):
        y_text = true_lams_df.iloc[-1, lam_idx] + y_offset
    
        ax.text(
            x_text,
            y_text,
            latex_symbol,
            color=line.get_color(),
            ha="left",
            va="center",
            clip_on=False
        )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        fig.savefig(
            TRUE_LAMS_PLOT_DIR / "true_lams_plot.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


#### TRUE TRANSITIONS RATE PLOT: CASE STUDY #3 ####


def generate_true_rates_plot_example_c(*,
        example_folder_name,
        n,
        true_lams_df,
        t_K: float = 3
    ):
    
    """
    Creates and saves the plot of the true transition rate L_1_2 for
    the third case study on piecewise-constant rates with an abrupt change.
    """
    
    latex_symbol = "$\\phi$"
    
    TRUE_LAMS_PLOT_DIR = (
        PLOTS_ROOT_FOLDER_DIR /
        f"{example_folder_name}"
    )
    TRUE_LAMS_PLOT_DIR.mkdir(exist_ok=True)
    
    K_plot = true_lams_df.shape[0] - 1
    time_series_plot = np.linspace(0, t_K, K_plot + 1)
    
    fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH_IN, 1.6))
    
    lam_idx = 0 # L_1_2
    lam = true_lams_df.columns[lam_idx]
    
    line, = ax.plot(
        time_series_plot,
        true_lams_df[lam],
        lw=2
    )
    
    ax.set_xlim(0 - 0.05, 3 + 0.05)
    ax.set_xlabel("time")
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20])
    
    # Label curve near just under its end
    x_text = time_series_plot[-1] - 0.2
    y_offset = -3.4
    
    y_text = true_lams_df.iloc[-1, lam_idx] + y_offset

    ax.text(
        x_text,
        y_text,
        latex_symbol,
        color=line.get_color(),
        ha="left",
        va="center",
        clip_on=False
    )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        fig.savefig(
            TRUE_LAMS_PLOT_DIR / "true_lams_plot.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


#### TRUE TRANSITIONS RATE PLOT: CASE STUDY #4 ####


def generate_true_rates_plot_case_study_4(*,
        example_folder_name,
        n,
        true_lams_df,
        t_K: float = 3
    ):
    
    """
    Creates and saves the plot of the true non-zero transition rates for
    the Case Study #4 on a high-dimensional state space.
    """
    
    # Select columns of non-zero transition rates
    true_lams_df = true_lams_df[["L_1_2", "L_2_3", "L_3_4", "L_4_5", "L_5_1"]]
    
    TRUE_LAMS_PLOT_DIR = (
        PLOTS_ROOT_FOLDER_DIR /
        f"{example_folder_name}"
    )
    TRUE_LAMS_PLOT_DIR.mkdir(exist_ok=True)
    
    K_plot = true_lams_df.shape[0] - 1
    time_series_plot = np.linspace(0, t_K, K_plot + 1)
    
    fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH_IN, 2))
    
    lines = []
    
    for lam_idx, lam in enumerate(true_lams_df.columns):
        
        p, q = get_p_q_from_lam_col_name(lam)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        line, = ax.plot(
            time_series_plot,
            true_lams_df[lam],
            lw=2,
            alpha=0.7
        )
        
        lines.append((line, latex_symbol, lam_idx))
    
    ax.set_xlim(-0.05, 3.55) # more space on right so labels fit
    ax.set_xlabel("time")
    ax.set_ylim(-0.1, 4.6)
    ax.set_yticks(np.arange(0, 5))
    
    
    for i, (line, latex_symbol, lam_idx) in enumerate(lines):
        
        if i == len(lines) - 1:
            x_text = time_series_plot[-1] - 0.6
            y_offset = 0.3
        else:
            x_text = time_series_plot[-1] + 0.06
            y_offset = -0.16
        
        y_text = true_lams_df.iloc[-1, lam_idx] + y_offset
        
        ax.text(
            x_text,
            y_text,
            latex_symbol,
            color=line.get_color(),
            ha="left",
            va="center",
            clip_on=False
        )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        fig.savefig(
            TRUE_LAMS_PLOT_DIR / "true_lams_plot.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


###### CASE STUDY 1 RUN RESULTS GENERATION ######


def generate_group_plots_for_case_study(*,
        case_study_num: int,
        group_num: int,
        start_index: int,
        runs_table_dictionary: dict,
        example_folder_name: str = None,
        gp: bool = False, # True only if Gaussian process runs
        bp_ymin: list = None,
        bp_ymax: list = None,
        y_ticks: list = None,
        K_plot: int = 5000,
        generate_true_rates_plot: bool = False,
    ):
    
    """
    Generate the plots of a group of runs for for a particular case study:
    - true transition rates plot (if applicable),
    - band plots and optional state proportion plots stacked vertically,
    - the data plot (rarely),
    - the ESS plot (rarely).
    
    Inputs
    ------
    gp: True if all runs are Gaussian process runs; False if all are not.
    K_plot: the number of steps between t_0=0 and t_K (inclusive) that the
        true rates are computed at for plotting.
    """
    
    if case_study_num not in CASE_STUDY_NUMS:
        raise ValueError(
            f"case_study_num must be one of {CASE_STUDY_NUMS}. "
            f"Currently, it is {case_study_num}."
        )
    
    if example_folder_name is None:
        example_folder_name = f"CTMC_CaseStudy{case_study_num}_Figs"
    
    PLOTS_FOLDER_DIR = (
        PLOTS_ROOT_FOLDER_DIR / 
        f"{example_folder_name}/Group_{group_num}"
    )
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    num_runs = len(runs_table_dictionary["N"])
    assert all(len(row) == num_runs for row in runs_table_dictionary.values())
    
    
    #### Define case study parameters ####
    
    if case_study_num == 1:
        
        n = 3
        t_K = None # Varies between runs
        
        if not gp:
            mu0    = np.array([1, 1, 1, 1, 5, 1])
            var0   = np.array([4, 4, 4, 4, 6, 4])
        else:
            mu     = np.array([1, 1, 1, 1, 5, 1])
    
    elif case_study_num == 2:
        
        n = 2
        t_K = 3
        
        if not gp:
            mu0  = np.array([1.8, 3.8])
            var0 = np.array([2,   2  ])
        else:
            mu   = np.array([1.8, 3.8])
    
    elif case_study_num == 3:
        
        n = 2
        t_K = 3
        
        if not gp:
            mu0  = np.array([1, 1])
            var0 = np.array([2, 2])
        else:
            mu   = np.array([1, 1])
        
    elif case_study_num == 4:
        
        n = 5
        t_K = 3
        
        if not gp:
            mu0  = np.repeat(2, 20)
            var0 = np.repeat(1, 20)
        else:
            mu   = np.repeat(2, 20)
    
    else:
        
        raise NotImplementedError(
            f"Case Study #{case_study_num} has not been implemented."
        )
    
    
    #### Generate and store the group results ####
    
    print(f"Generating group results: Group {group_num}")
    print("=================================================")
    print()
    
    group_results = {}
    run_letters = []
    
    for i in range(num_runs):
        
        run_letter = map_num_to_letter(start_index + i)
        run_letters.append(run_letter)
        
        N = runs_table_dictionary["N"][i]
        delta_t = runs_table_dictionary["dt"][i]
        K = runs_table_dictionary["K"][i]
        J = runs_table_dictionary["J"][i]
        
        if t_K is not None:
            assert K * delta_t == t_K
        
        if not gp:
            TV = runs_table_dictionary["TV"][i]
            C = runs_table_dictionary["C"][i]
            ssm_params = {"mu0": mu0, "var0": var0, "TV": TV, "C": C}
        else:
            kappa = runs_table_dictionary["kappa"][i]
            sigma2 = runs_table_dictionary["sigma2"][i]
            ssm_params = {"mu": mu, "kappa": kappa, "sigma2": sigma2}
        
        run_results = generate_run_results(
            N = N,
            n = n,
            delta_t = delta_t,
            K = K,
            J = J,
            ssm_params = ssm_params,
            case_study_num = case_study_num,
            gp = gp
        )
        
        group_results[run_letter] = run_results
    
    ## Store run letters in runs_table_dictionary ##
    
    runs_table_dictionary["run_letter"] = run_letters
    
    ## Create true rates DF for plotting true rates in band plots ##
    
    if case_study_num == 1:
        true_rates_df_plot = make_true_rates_dataframe(
            true_rates = simulate_example_a(K=K_plot),
            n = n
        )
    
    elif case_study_num == 2:
        true_rates_df_plot = make_true_rates_dataframe(
            true_rates = simulate_sine_rates_n2(K=K_plot),
            n = n
        )
        
    elif case_study_num == 3:
        true_rates_df_plot = make_true_rates_dataframe(
            true_rates = simulate_example_c(K=K_plot),
            n = n
        )
        
    elif case_study_num == 4:
        true_rates_df_plot = make_true_rates_dataframe(
            true_rates = simulate_case_study_4(K=K_plot),
            n = n
        )
        
    else:
        raise NotImplementedError(
            f"Case Study #{case_study_num} has not been implemented."
        )
    
    
    #### True transition rate plots ####
    
    if generate_true_rates_plot:
        
        if case_study_num == 2:
            generate_true_rates_plot_example_b(
                example_folder_name = example_folder_name,
                n = n,
                true_lams_df = true_rates_df_plot,
            )
        
        elif case_study_num == 3:
            generate_true_rates_plot_example_c(
                example_folder_name = example_folder_name,
                n = n,
                true_lams_df = true_rates_df_plot,
            )
            
        elif case_study_num == 4:
            generate_true_rates_plot_case_study_4(
                example_folder_name = example_folder_name,
                n = n,
                true_lams_df = true_rates_df_plot,
            )
            
        else:
            raise NotImplementedError(
                f"Case Study #{case_study_num} has not been implemented."
            )
        
        print("Created the true transition rates plot.")
        print()
    
    
    #### Stack all plots vertically for Case Study #1 and when J == 1 ####
    
    if (
            case_study_num == 1 and
            num_runs == 1 and
            runs_table_dictionary["J"][0] == 1
        ):
        
        run_letter = runs_table_dictionary["run_letter"][0]
        
        stack_all_plots_vertically(
            n = n,
            time_series = group_results[run_letter]["time_series"],
            true_rates_df_plot = true_rates_df_plot,
            data = group_results[run_letter]["data"],
            ds_boot = group_results[run_letter]["ds_boot"],
            plots_folder_dir = PLOTS_FOLDER_DIR,
        )
    
    
    #### Plot band plots of 2 or 4 runs for Case Study #3 ####
    
    elif case_study_num == 3:
        
        if num_runs != 2 and num_runs != 4:
            raise ValueError(
                f"This plotting code expects 2 or 4 runs, but got {num_runs}."
            )
        
        band_plots_cs3(
            num_runs = num_runs,
            run_letters = runs_table_dictionary["run_letter"],
            group_results = group_results,
            true_rates_df_plot = true_rates_df_plot,
            n = n,
            bp_ymin = bp_ymin,
            bp_ymax = bp_ymax,
            y_ticks = y_ticks,
            plots_folder_dir = PLOTS_FOLDER_DIR,
        )
    
    
    #### BAND PLOTS & STATE PROPORTION PLOTS STACKED VERTICALLY ####
    
    else:
    
        if case_study_num == 1:
            
            band_nrows = 2
            band_ncols = 3
            
        elif case_study_num == 2:
            
            band_nrows = 1
            band_ncols = 2
            
        elif case_study_num == 4:
            
            band_nrows = 2
            band_ncols = 3
            
            lams_to_plot = [
                "L_1_2", "L_2_3", "L_3_4", "L_4_5", "L_5_1", # Non-zero rates
                "L_3_1" # Zero rate
            ]
            
            true_rates_df_plot = true_rates_df_plot[lams_to_plot]
            
            for run_letter in runs_table_dictionary["run_letter"]:
                group_results[run_letter]["ds_boot"] = (
                    group_results[run_letter]["ds_boot"]
                    .sel(lam=lams_to_plot)
                )
            
        else:
            
            raise NotImplementedError(
                "Vertically stacked band plots has not been implemented "
                f"for Case Study #{case_study_num}."
            )
        
        
        band_plots_and_optional_state_proportions(
            case_study_num = case_study_num,
            num_runs = num_runs,
            run_letters = runs_table_dictionary["run_letter"],
            n = n,
            true_rates_df_plot = true_rates_df_plot,
            group_results = group_results,
            band_nrows = band_nrows,
            band_ncols = band_ncols,
            plot_state_proportions_bool = runs_table_dictionary["pbc"],
            bp_ymin = bp_ymin,
            bp_ymax = bp_ymax,
            y_ticks = y_ticks,
            plots_folder_dir = PLOTS_FOLDER_DIR,
        )
    
    
    #### SEPARATE DATA & ESS PLOTS ####
    
    if case_study_num == 1:
    
        for i in range(num_runs):
        
            run_J = runs_table_dictionary["J"][i]
            
            if run_J == 8:
                
                run_letter = runs_table_dictionary["run_letter"][i]
                
                plot_data_ess_special_case_cs1(
                    time_series = group_results[run_letter]["time_series"],
                    data = group_results[run_letter]["data"],
                    ess = group_results[run_letter]["ds_boot"]["ESS"],
                    plots_folder_dir = PLOTS_FOLDER_DIR
                )
    
    
    print()
    print(f"Finished generating group results: Group {group_num}")
    print("=================================================")
    print()
    print()






if __name__ == "__main__":
    pass






###### Old plotting code ######



## Plot data (if J not too large) ##

# if J <= 10:
#     data_plot = np.vstack(data)
    
#     fig, axes = plt.subplots(
#         nrows=J,
#         ncols=1,
#         sharex=True,
#         figsize=(8, 3)
#     )
#     fig.suptitle("RW states over time", fontsize=14)
    
#     # Ensure axes is always iterable (important if J == 1)
#     if J == 1:
#         axes = [axes]
    
#     for j in range(J):
#         axes[j].plot(k_series, data_plot[:, j])
#         axes[j].set_ylabel(f"RW #{j+1}")
#         axes[j].grid(True)
    
#     axes[-1].set_xlabel("k")
#     plt.tight_layout()
#     if SAVE_PLOTS:
#         op_num = get_option_num_for_transition_dist(px_var_flag)
#         folder_path = Path(PLOT_ROOT_FOLDER_NAME + "/"
#                            + f"{EXAMPLE_FOLDER_NAME}/Option {op_num}; J={J}")
#         folder_path.mkdir(parents=True, exist_ok=True)
#         image_name = f"RW_States_J{J}_Op{op_num}.png"
#         plt.savefig(folder_path / image_name,
#                     bbox_inches='tight')
#         plt.close("all")
#     else:
#         plt.show()
# else:
#     print(f"Too many random walkers to plot: {J} RWs.")



## Band plots: 5th quantile, median, 95th quantile ##

# for lam_idx, lam in enumerate(true_lams.columns):
#     p, q = lams_idx_to_gen_pos(lam_idx, n)
#     latex_symbol, rate_name_img = get_latex_rate_symbol(p, q)
    
#     median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
#     lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
#     uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
#     plt.plot(k_series, true_lams[lam].values, label=f"True {latex_symbol}",
#              color='red', alpha=0.7)
#     plt.plot(median, color="green",
#              label="PF mean", alpha=0.7)
#     plt.fill_between(k_series, 
#                      y1=lq, 
#                      y2=uq, 
#                      color="green", alpha=0.3)
#     plt.legend()
#     plt.xlabel("k")
#     plt.ylabel(f"Value of {latex_symbol}")
#     plt.title(f"Boot PF band plot (quantiles): {latex_symbol} | "
#               + f"J={J}; N={N}; $\Delta t$={delta_t}; C={C}")
#     if SAVE_PLOTS:
#         folder_path = Path(FOLDER_PATH_STR)
#         folder_path.mkdir(parents=True, exist_ok=True)
#         image_name = f"Quantiles_{rate_name_img}_J{J}_Op{op_num}.png"
#         plt.savefig(folder_path / image_name,
#                     bbox_inches='tight')
#         plt.close("all")
#     else:
#         plt.show()



## KDE AND PAIRWISE SCATTER PLOTS ##

# if PLOT_EXTRAS:
    
#     ## Choose some k between 0 and K+1 (inclusive) ##
    
#     k = np.random.randint(K+1)
    
    
#     ## KDE for each lam (uses weights of particles) ##
    
#     for lam in true_lams.columns:
#         fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 6))
#         sns.kdeplot(x=(ds_boot["X"].sel({'lam': lam, 'k': k})
#                        .values.reshape(-1)),
#                     weights=ds_boot["W"].sel({'k': k}).values.reshape(-1),
#                     ax=ax, fill=True,
#                     color="skyblue", label="Boot")
#         ax.axvline(x=true_lams.loc[k][lam], color='red', linestyle=':',
#                    linewidth=1.5, label='True state')
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Density")
#         ax.set_title(f"Boot Filtering Dist. @ k = {k}: {lam}")
#         ax.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.show()
    
    
#     ## Pairwise scatter plots of particles (not using weights) ##
    
#     plot_df = (
#         ds_boot["X"].sel(k=k)
#         .to_pandas() # index: particle, columns: lambda
#         .reset_index(drop=True)
#     )
    
#     sns.pairplot(
#         plot_df,
#         plot_kws={"alpha": 0.5, "s": 15}
#     )
#     plt.suptitle(f"Pairwise scatter at k = {k}: Boot PF", y=1.02)
#     plt.show()



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

