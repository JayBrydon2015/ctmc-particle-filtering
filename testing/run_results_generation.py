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
    get_gamma_params_from_mean_var,
    CTMC,
    GP_CTMC
)

from rates_simulation.true_rates_simulation_funtions import (
    simulate_example_a, simulate_sine_rates_n2,
    simulate_data, simulate_data_manually_example_a
)


###### CONSTANTS ######

SAVE_PLOTS  = True  # Save plots to a folder (True) or show them (False)
PLOT_EXTRAS = False # Plot extra stuff (True rates, KDEs and PW scatter plots)

PLOTS_ROOT_FOLDER_DIR = Path(__file__).parent / "generated_plots"
PLOTS_ROOT_FOLDER_DIR.mkdir(exist_ok=True)

EXAMPLE_LETTERS = ["a", "b", "c"]

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


def get_latex_rate_symbol_ex_a(p, q):
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
        return f"$\\lambda^{{{p} \\to {q}}}$", f"L{p}{q}"


def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate used in the image file
        name.
    """
    return f"$\\lambda^{{{p} \\to {q}}}$", f"L{p}{q}"


def weighted_quantile(values, weights, quantiles):
    """
    Calculates given quantiles using values and weights.
    
    -- Inputs --
    values: (N,)
    weights: (N,)
    quantiles: array-like in [0,1]
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


def make_true_lams_dataframe(*, true_states_plot, n, time_series_plot):
    """ Creates a pandas data frame storing the true rates. """
    
    lams_gen_positions = [
        lams_idx_to_gen_pos(i, n)
        for i in range(true_states_plot[0].shape[1])
    ]
    
    true_lams_df = pd.DataFrame(
        np.stack([true_state.reshape(-1) for true_state in true_states_plot]),
        columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
        index=time_series_plot
    ).rename_axis('time')
    
    return true_lams_df


def run_particle_filter(*,
        ctmc_ssm,
        data,
        N,
        true_lams_df, 
        k_series,
        exp_X = False,
        qs = None
    ):
    
    """ Runs the bootstrap particle filter and stores the results
        into an xarray.Dataset
    
        If exp_X is True, exponentiate the particles' values.
        
        If quantiles is None, calculate and store the quantiles 5%, 50%, and
        95% of the filtering ensembles in the Dataset.
    """
    
    if qs is None:
        qs = np.array([0.05, 0.5, 0.95]) # 95% interval & median
    
    
    ###### Run the bootstrap particle filter ######
    
    fk_boot = augssm.AugmentedBootstrap(ssm=ctmc_ssm, data=data)
    pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                            store_history=True, collect=[Moments()])
    print("Beginning the bootstrap particle filter.")
    pf_boot.run()
    print("Bootstrap particle filter finished.")
    
    
    ###### Store lambda particles and weights in an xarray.Dataset ######
    
    ds_boot = xr.Dataset({
        
        'X': xr.DataArray(
            np.stack([pf_boot.hist.X[k] for k in k_series]),
            dims=("k", "particle", "lam"),
            coords={
                "k": k_series,
                "lam": true_lams_df.columns.values,
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
    ).assign_coords(quantile=qs)
    
    return ds_boot


def generate_run_results(*,
        N, n, ssm_params: dict, delta_t, K, J, C, example, gp: bool,
        K_plot: int = 5000
    ):
    
    """ Generate the results of a run in a group for one of the examples.
        Return the results as a dictionary.
    
    Inputs
    ------
    ctmc_params: dictionary of CTMC SSM parameters, specifically those that
        differ between the Gamma transitions SSM and the Gaussian process SSM.
    example: one of "a", "b", "c", "d", depending on which example.
    gp: True if modelling log-rates in the hidden process; False otherwise.
    K_plot: the number of steps between t_0=0 and t_K (inclusive) that the
        true rates are computed at for plotting.
    """
    
    if example not in EXAMPLE_LETTERS:
        raise ValueError(
            f"'example' must be one of {EXAMPLE_LETTERS}. "
            f"Currently, it is {example}."
        )
    
    k_series    = np.arange(K + 1)   # [0, 1, ..., K]
    time_series = delta_t * k_series # [t_0, t_1, ..., t_K], given t_0 = 0
    t_K = time_series[-1]
    
    exp_X = True if gp else False
    
    ## Unpack the CTMC parameters ##
    
    if not gp:
        a0 = ssm_params["a0"]
        b0 = ssm_params["b0"]
        TV = ssm_params["TV"]
    else:
        mu0 = ssm_params["mu0"]
        scale0 = ssm_params["scale0"]
        l = ssm_params["l"]
    
    ## Define the parameters not altered between runs ##
    ## but unique to the example ##
    
    if example == "a":
        y_init = np.array([0 for _ in range(J)]) # RWs initial config y_{-1}
        true_states = simulate_example_a(K=K) # Used in PF
        true_states_plot = simulate_example_a(K=K_plot)
        
    elif example == "b":
        y_init = None
        true_states = simulate_sine_rates_n2(K=K) # Used in PF
        true_states_plot = simulate_sine_rates_n2(K=K_plot)
        
    elif example == "c":
        y_init = None
        true_states = None # Do this later
        
    else:
        raise NotImplementedError
    
    ## Store true rates in Pandas dataframe ##
    
    true_lams_df = make_true_lams_dataframe(
        true_states_plot = true_states_plot,
        n = n,
        time_series_plot = np.linspace(0, t_K, K_plot + 1)
    )
    
    ## Create CTMC SSM ##
    
    if not gp:
    
        ctmc_ssm = CTMC(
            n = n,
            J = J,
            delta_t = delta_t,
            TV = TV,
            C = C,
            a0 = a0,
            b0 = b0,
            y_init = y_init,
            px_verbose = True
        )
    
    else:
    
        ctmc_ssm = GP_CTMC(
            n = n,
            J = J,
            delta_t = delta_t,
            l = l,
            C = C,
            mu0 = mu0,
            scale0 = scale0,
            y_init = y_init,
            px_verbose = True
        )
    
    if y_init is None:
        y_init = ctmc_ssm.y_init
    
    ## Simulate data ##
    
    if example == "a" and J == 1 and K == 300:
        
        # Manual simulation of the data only occurs in example A
        data = simulate_data_manually_example_a()
        
    else:
        
        data = simulate_data(
            true_rates=true_states,
            n=n,
            J=J,
            delta_t=delta_t,
            y_init=y_init
        )
    
    ## Run particle filter ##
    
    ds_boot = run_particle_filter(
        ctmc_ssm = ctmc_ssm,
        data = data,
        N = N,
        true_lams_df = true_lams_df, 
        k_series = k_series,
        exp_X = exp_X,
    )
    
    return {
        "k_series": k_series,
        "time_series": time_series,
        "true_lams_df": true_lams_df,
        "data": data,
        "ds_boot": ds_boot
    }


###### EXAMPLE A RUN RESULTS GENERATION ######


def generate_group_plots_example_a(*,
        group_num,
        start_index,
        runs_table_dictionary,
        example_folder_name,
        gp=False, # True only if Gaussian process runs
        bp_ymin=None,
        bp_ymax=None,
        y_ticks=None
    ):
    
    """
    Generate the plots of a group of runs for Example A:
    - the data plot,
    - the ESS plot,
    - the band plots, and
    - any extras if PLOT_EXTRAS is True and SAVE_PLOTS is False. These plots
      will be showed and not saved.
    """
    
    n = 3
    
    if not gp:
        mu0    = np.array([1, 1, 1, 1, 5, 1]) # For P(lams_0), a vague prior
        var0   = np.array([4, 4, 4, 4, 6, 4]) # For P(lams_0), a vague prior
        a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
    else:
        mu0    = np.array([1, 1, 1, 1, 5, 1]) # For P(llams0), a vague prior
        scale0 = np.array([1, 1, 1, 1, 1, 1]) # For P(llams0), a vague prior
    
    PLOTS_FOLDER_DIR = (
        PLOTS_ROOT_FOLDER_DIR / 
        f"{example_folder_name}/Group_{group_num}"
    )
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating group results: Group {group_num}")
    print("=================================================")
    print()
    
    
    num_runs = len(runs_table_dictionary["N"])
    assert all(len(row) == num_runs for row in runs_table_dictionary.values())
    
    ## Generate and store the group results ##
    
    group_results = {}
    run_letters = []
    
    for i in range(num_runs):
        
        run_letter = map_num_to_letter(start_index + i)
        run_letters.append(run_letter)
        
        N = runs_table_dictionary["N"][i]
        delta_t = runs_table_dictionary["dt"][i]
        K = runs_table_dictionary["K"][i]
        J = runs_table_dictionary["J"][i]
        C = runs_table_dictionary["C"][i]
        
        if not gp:
            TV = runs_table_dictionary["TV"][i]
            ssm_params = {"a0": a0, "b0": b0, "TV": TV}
        else:
            l = runs_table_dictionary["l"][i]
            ssm_params = {"mu0": mu0, "scale0": scale0, "l": l}
        
        run_results = generate_run_results(
            N = N,
            n = n,
            ssm_params = ssm_params,
            delta_t = delta_t,
            K = K,
            J = J,
            C = C,
            example = "a",
            gp = gp
        )
        
        group_results[run_letter] = run_results
    
    ## Store run letters in runs_table_dictionary ##
    
    runs_table_dictionary["run_letter"] = run_letters
    
    
    ###### Stack all plots vertically for when J == 1 ######
    
    if num_runs == 1 and runs_table_dictionary["J"][0] == 1:
        
        run_letter = runs_table_dictionary["run_letter"][0]
        
        time_series = group_results[run_letter]["time_series"]
        true_lams_df = group_results[run_letter]["true_lams_df"]
        data = group_results[run_letter]["data"]
        ds_boot = group_results[run_letter]["ds_boot"]
        
        t_K = time_series[-1]
        
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
        
        for lam_idx, lam in enumerate(true_lams_df.columns):
            
            if lam_idx in {1, 5}:
                continue
            
            ax = band_axes[band_ax_idx]
            
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            
            latex_symbol, _ = get_latex_rate_symbol_ex_a(p, q)
            
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
                true_lams_df.index,
                true_lams_df[lam].values,
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
                PLOTS_FOLDER_DIR / "J1_all_plots_combined.pdf",
                format="pdf",
                bbox_inches='tight',
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    ###### BAND PLOTS & BIN COUNTS OVER TIME PLOT COMBINED ######
    
    elif num_runs == 1 and runs_table_dictionary["pbc"][0]:
    
        run_letter = runs_table_dictionary["run_letter"][0]
        
        time_series = group_results[run_letter]["time_series"]
        true_lams_df = group_results[run_letter]["true_lams_df"]
        data = group_results[run_letter]["data"]
        ds_boot = group_results[run_letter]["ds_boot"]
        
        t_K = time_series[-1]
        
        counts = np.array([
            np.bincount(data_k.reshape(-1), minlength=n)
            for data_k in data
        ])
        
        proportions = (
            counts /
            np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
        )
        
        # One figure:
        # top 2 rows = 2x3 band plots
        # bottom row = full-width state proportions plot
        fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.2))

        outer_gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[2.0, 1.15],
            hspace=0.3
        )
        
        band_gs = outer_gs[0].subgridspec(
            nrows=2,
            ncols=3,
            hspace=0.11,
            wspace=0.32
        )
        
        axes = []
        for row in range(2):
            for col in range(3):
                share_ax = axes[0] if axes else None
                axes.append(
                    fig.add_subplot(band_gs[row, col], sharex=share_ax)
                )
        
        pbc_ax = fig.add_subplot(outer_gs[1])
        
        # =========================
        # Band plots
        # =========================
        for lam_idx, lam in enumerate(true_lams_df.columns):
            ax = axes[lam_idx]
    
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            latex_symbol, rate_name_img = get_latex_rate_symbol_ex_a(p, q)
    
            median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
            lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
            uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    
            ax.plot(
                true_lams_df.index,
                true_lams_df[lam].values,
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
    
            ax.set_ylabel(latex_symbol, labelpad=2.1)
    
            if bp_ymin is not None and bp_ymax is not None:
                ax.set_ylim(
                    bottom=bp_ymin[lam_idx] - 0.1,
                    top=bp_ymax[lam_idx] + 0.1
                )
    
            if y_ticks is not None:
                ax.set_yticks(y_ticks[lam_idx])
    
            # Only show legend once
            # if lam_idx == 2:
            #     ax.legend(loc=1, framealpha=0.5)
    
        for ax in axes:
            ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
    
        # Hide x tick labels on top row of band plots
        for ax in axes[:3]:
            ax.tick_params(axis="x", labelbottom=False)
    
        # Bottom row of band plots gets x-axis labels
        for ax in axes[3:]:
            ax.set_xlabel("time")
        
        # =========================
        # State proportions plot
        # =========================
        labels = [f"{value + 1}" for value in range(n)]
    
        pbc_ax.stackplot(
            time_series,
            proportions.T,
            labels=labels,
            alpha=0.7,
            edgecolor='white', linewidth=1
        )
    
        pbc_ax.set_xlabel("time")
        pbc_ax.set_xlim(0, t_K)
        pbc_ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
        pbc_ax.set_ylabel("Proportion")
        pbc_ax.set_ylim(0, 1)
        pbc_ax.set_yticks([0.0, 0.5, 1.0])
        pbc_ax.legend(
            ncols=3,
            title="State",
            loc="lower left",
            framealpha=0.5
        )
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "band_plots_and_state_proportions.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    ###### STACK BAND PLOTS OF EACH RUN ON TOP OF EACH OTHER ######
    
    else:
        
        fig = plt.figure(figsize=(TEXTWIDTH_IN, 3 * num_runs))

        # Controls spacing BETWEEN run-blocks
        outer_gs = fig.add_gridspec(
            nrows=num_runs,
            ncols=1,
            hspace=0.1
        )
        
        all_axes = []
        
        for run_idx in range(num_runs):
            run_letter = runs_table_dictionary["run_letter"][run_idx]
        
            time_series = group_results[run_letter]["time_series"]
            true_lams_df = group_results[run_letter]["true_lams_df"]
            ds_boot = group_results[run_letter]["ds_boot"]
        
            t_K = time_series[-1]
        
            # Controls spacing WITHIN this 2x3 block
            block_gs = outer_gs[run_idx].subgridspec(
                nrows=2,
                ncols=3,
                hspace=0.1,
                wspace=0.3
            )
        
            run_axes = np.empty((2, 3), dtype=object)
        
            for row in range(2):
                for col in range(3):
                    if row == 0 and col == 0:
                        ax = fig.add_subplot(block_gs[row, col])
                    else:
                        ax = fig.add_subplot(block_gs[row, col])
                    run_axes[row, col] = ax
        
            all_axes.append(run_axes)
        
            flat_axes = run_axes.flatten()
        
            for lam_idx, lam in enumerate(true_lams_df.columns):
                ax = flat_axes[lam_idx]
        
                p, q = lams_idx_to_gen_pos(lam_idx, n)
                latex_symbol, rate_name_img = get_latex_rate_symbol_ex_a(p, q)
        
                median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
                lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
                uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
        
                ax.plot(
                    true_lams_df.index,
                    true_lams_df[lam].values,
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
        
                ax.set_ylabel(latex_symbol, labelpad=2.1)
                
                if bp_ymin is not None and bp_ymax is not None:
                    ax.set_ylim(
                        bottom=bp_ymin[lam_idx] - 0.1,
                        top=bp_ymax[lam_idx] + 0.1
                    )
                
                if y_ticks is not None:
                    ax.set_yticks(y_ticks[lam_idx])
        
                ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
        
                # if run_idx == 0 and lam_idx == 2:
                #     ax.legend(loc=1, framealpha=0.5)
        
            # Top row of this block: no x tick labels
            for ax in run_axes[0, :]:
                ax.tick_params(axis="x", labelbottom=False)
        
            # Bottom row of this block: yes x tick labels
            for ax in run_axes[1, :]:
                ax.tick_params(axis="x", labelbottom=True)
        
            # Only the VERY LAST block gets the x-axis label
            if run_idx == num_runs - 1:
                for ax in run_axes[1, :]:
                    ax.set_xlabel("time")
            else:
                for ax in run_axes[1, :]:
                    ax.set_xlabel("")
        
            # Run label for the whole block
            top_left_ax = run_axes[0, 0]
            bottom_left_ax = run_axes[1, 0]
            
            pos_top = top_left_ax.get_position()
            pos_bottom = bottom_left_ax.get_position()
            
            top_y_center = 0.5 * (pos_top.y0 + pos_top.y1)
            bottom_y_center = 0.5 * (pos_bottom.y0 + pos_bottom.y1)
            
            if num_runs == 2:
                y_center = (
                    0.5 * (top_y_center + bottom_y_center)
                    + RUN_LABEL_Y_OFFSETS_2_RUNS.get(run_idx, 0.0)
                )
            elif num_runs == 3:
                y_center = (
                    0.5 * (top_y_center + bottom_y_center)
                    + RUN_LABEL_Y_OFFSETS_3_RUNS.get(run_idx, 0.0)
                )
            else:
                y_center = (0.5 * (top_y_center + bottom_y_center))
            
            x_left = pos_top.x0 - 0.025
            
            fig.text(
                x_left,
                y_center,
                f"Run {run_letter}",
                ha="right",
                va="center",
                rotation=90,
                fontsize=FONT_SIZE
            )
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        
        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.08)
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "group_band_plots.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    ###### SEPARATE DATA & ESS PLOTS ######
    
    for i in range(num_runs):
    
        run_J = runs_table_dictionary["J"][i]
        
        if run_J == 8:
            
            run_letter = runs_table_dictionary["run_letter"][i]
            
            time_series = group_results[run_letter]["time_series"]
            data = group_results[run_letter]["data"]
            ds_boot = group_results[run_letter]["ds_boot"]
            
            t_K = time_series[-1]
            
            data_plot = np.vstack(data)
            data_plot += 1  # '0' -> State 1, etc
            
            n_walkers = data_plot.shape[1]
            
            if n_walkers != 8:
                raise ValueError(
                    f"Expected 8 random walkers for run {run_letter}, "
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
            
            ess_ax.plot(time_series, ds_boot["ESS"], color="red", lw=1)
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
                    PLOTS_FOLDER_DIR / "J8_data_ess.pdf",
                    format="pdf",
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()
    
    
    print()
    print(f"Finished generating group results: Group {group_num}")
    print("=================================================")
    print()
    print()


###### EXAMPLE B RUN RESULTS GENERATION ######


def generate_true_rates_plot_example_b(*,
        example_folder_name,
        n,
        k_series,
        true_lams_df,
    ):
    
    """
    Creates and saves the plot of the true transition rates for
    Example B.
    """
    
    TRUE_LAMS_PLOT_DIR = (
        PLOTS_ROOT_FOLDER_DIR /
        f"{example_folder_name}"
    )
    
    fig, ax = plt.subplots(
        figsize=(TEXTWIDTH_IN, 2.5)
    )
    
    for lam_idx, lam in enumerate(true_lams_df.columns):
        
        p, q = lams_idx_to_gen_pos(lam_idx, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        ax.plot(
            true_lams_df.index,
            true_lams_df[lam],
            label=latex_symbol
        )
    
    ax.set_xlabel("k")
    ax.set_ylabel("Value")
    ax.set_ylim(-0.1, 4.3)
    ax.set_yticks(np.arange(0, 5))
    ax.legend(loc="lower right")
    
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


def generate_group_plots_example_b(*,
        group_num,
        start_index,
        runs_table_dictionary,
        example_folder_name,
        gp=False, # True only if Gaussian process runs
        bp_ymin=None,
        bp_ymax=None,
        y_ticks=None,
        generate_true_rates_plot=False,
    ):
    
    """
    Generate the plots of a group of runs for Example B:
    - the data plot,
    - the ESS plot,
    - the band plots, and
    - any extras if PLOT_EXTRAS is True and SAVE_PLOTS is False. These plots
      will be showed and not saved.
    """
    
    n = 2
    
    if not gp:
        mu0  = np.array([1.8, 3.8])
        var0 = np.array([2,   2  ])
        a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
    else:
        mu0  = np.array([1.8, 3.8])
        scale0 = np.array([1, 1])
    
    PLOTS_FOLDER_DIR = (
        PLOTS_ROOT_FOLDER_DIR / 
        f"{example_folder_name}/Group_{group_num}"
    )
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating group results: Group {group_num}")
    print("=================================================")
    print()
    
    
    num_runs = len(runs_table_dictionary["N"])
    assert all(len(row) == num_runs for row in runs_table_dictionary.values())
    
    ## Generate and store the group results ##
    
    group_results = {}
    run_letters = []
    
    for i in range(num_runs):
        
        run_letter = map_num_to_letter(start_index + i)
        run_letters.append(run_letter)
        
        N = runs_table_dictionary["N"][i]
        delta_t = runs_table_dictionary["dt"][i]
        K = runs_table_dictionary["K"][i]
        J = runs_table_dictionary["J"][i]
        C = runs_table_dictionary["C"][i]
        
        if not gp:
            TV = runs_table_dictionary["TV"][i]
            ssm_params = {"a0": a0, "b0": b0, "TV": TV}
        else:
            l = runs_table_dictionary["l"][i]
            ssm_params = {"mu0": mu0, "scale0": scale0, "l": l}
        
        run_results = generate_run_results(
            N = N,
            n = n,
            ssm_params = ssm_params,
            delta_t = delta_t,
            K = K,
            J = J,
            C = C,
            example = "b",
            gp = gp
        )
        
        group_results[run_letter] = run_results
    
    ## Store run letters in runs_table_dictionary ##
    
    runs_table_dictionary["run_letter"] = run_letters
    
    
    ###### Generate plots ######
    
    ## TRUE RATES PLOT ##
    
    if generate_true_rates_plot:
        
        generate_true_rates_plot_example_b(
            example_folder_name = example_folder_name,
            n = n,
            k_series = group_results["k_series"],
            true_lams_df = group_results["true_lams_df"],
        )
        
        # Stop the function after generating true rates plot.
        # So, this function will be called separately to when
        # we wish to create the other plots.
        return
    
    ## BAND PLOTS OPTIONALLY WITH STATE PROPORTIONS PLOTS ##
    
    num_runs = len(runs_table_dictionary["run_letter"])

    # Give runs with pbc a bit more height
    outer_height_ratios = [
        1.8 if runs_table_dictionary["pbc"][i] else 1.0
        for i in range(num_runs)
    ]
    
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.2 * sum(outer_height_ratios)))
    
    outer_gs = fig.add_gridspec(
        nrows=num_runs,
        ncols=1,
        height_ratios=outer_height_ratios,
        hspace = 0.3 if any(
            runs_table_dictionary["pbc"][i]
            for i in range(num_runs)
        ) else 0.38 # spacing BETWEEN run blocks
    )
    
    # Store axes for later run-label placement
    run_block_axes = []
    
    for run_idx in range(num_runs):
    
        run_letter = runs_table_dictionary["run_letter"][run_idx]
        plot_pbc = runs_table_dictionary["pbc"][run_idx]
    
        time_series = group_results[run_letter]["time_series"]
        true_lams_df = group_results[run_letter]["true_lams_df"]
        data = group_results[run_letter]["data"]
        ds_boot = group_results[run_letter]["ds_boot"]
    
        t_K = time_series[-1]
    
        # ----------------------------------
        # Inner layout for this run block
        # ----------------------------------
        if plot_pbc:
            block_gs = outer_gs[run_idx].subgridspec(
                nrows=2,
                ncols=2,
                height_ratios=[1.0, 1.0],
                hspace=0.32,   # spacing BETWEEN band plots row and pbc plot
                wspace=0.24
            )
        else:
            block_gs = outer_gs[run_idx].subgridspec(
                nrows=1,
                ncols=2,
                wspace=0.24
            )
    
        # ----------------------------------
        # Band plot axes
        # ----------------------------------
        band_axes = []
        for col in range(2):
            share_ax = band_axes[0] if band_axes else None
            band_axes.append(
                fig.add_subplot(block_gs[0, col], sharex=share_ax)
            )
    
        # ----------------------------------
        # Plot the 2 band plots
        # ----------------------------------
        # Assumes true_lams_df has exactly 2 columns for this example
        for lam_idx, lam in enumerate(true_lams_df.columns):
            ax = band_axes[lam_idx]
    
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            latex_symbol, _ = get_latex_rate_symbol_ex_a(p, q)
    
            median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
            lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
            uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    
            ax.plot(
                true_lams_df.index,
                true_lams_df[lam].values,
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
    
            ax.set_ylabel(latex_symbol, labelpad=2.1)
            ax.set_xlim(0, t_K)
            ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
    
            if bp_ymin is not None and bp_ymax is not None:
                ax.set_ylim(
                    bottom=bp_ymin[lam_idx] - 0.1,
                    top=bp_ymax[lam_idx] + 0.1
                )
    
            if y_ticks is not None:
                ax.set_yticks(y_ticks[lam_idx])
    
        # Optional: only one legend total
        # if run_idx == 0:
        #     band_axes[1].legend(loc="upper right", framealpha=0.5)
    
        # ----------------------------------
        # pbc/state proportions plot
        # ----------------------------------
        pbc_ax = None
    
        if plot_pbc:
            pbc_ax = fig.add_subplot(block_gs[1, :], sharex=band_axes[0])
    
            counts = np.array([
                np.bincount(data_k.reshape(-1), minlength=n)
                for data_k in data
            ])
    
            proportions = (
                counts /
                np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
            )
    
            labels = [f"{value + 1}" for value in range(n)]
    
            pbc_ax.stackplot(
                time_series,
                proportions.T,
                labels=labels,
                alpha=0.7,
                edgecolor="white",
                linewidth=1
            )
    
            pbc_ax.set_ylabel("Proportion")
            pbc_ax.set_xlabel("time")
            pbc_ax.set_xlim(0, t_K)
            pbc_ax.set_xticks([0, t_K / 3, 2 * t_K / 3, t_K])
            pbc_ax.set_ylim(0, 1)
            pbc_ax.set_yticks([0.0, 0.5, 1.0])
    
            # Show state legend only once if you want
            if run_idx == num_runs - 1:
                pbc_ax.legend(
                    ncols=n,
                    title="State",
                    loc="lower left",
                    framealpha=0.5
                )
            
            # for ax in band_axes:
            #     ax.tick_params(axis="x", labelbottom=False)
            #     ax.set_xlabel("")
            
        else:
            # No pbc: band plots are the bottom of the block
            for ax in band_axes:
                ax.set_xlabel("time")
    
        run_block_axes.append({
            "run_letter": run_letter,
            "band_axes": band_axes,
            "pbc_ax": pbc_ax
        })
    
    # ----------------------------------
    # Global formatting
    # ----------------------------------
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    
    # fig.subplots_adjust(
    #     left=0.16,
    #     right=0.98,
    #     top=0.98,
    #     bottom=0.07
    # )
    
    # ----------------------------------
    # Add run labels on the left
    # ----------------------------------
    RUN_LABEL_X = 0.045
    
    for run_idx, block_info in enumerate(run_block_axes):
        run_letter = block_info["run_letter"]
        band_axes = block_info["band_axes"]
        pbc_ax = block_info["pbc_ax"]
    
        top_left_ax = band_axes[0]
    
        if pbc_ax is None:
            # Centre across the bandplots row only
            pos = top_left_ax.get_position()
            y_center = 0.5 * (pos.y0 + pos.y1)
        else:
            # Centre across the full block: top of bandplots row to bottom of pbc
            pos_top = top_left_ax.get_position()
            pos_bottom = pbc_ax.get_position()
            y_center = 0.55 * (pos_top.y1 + pos_bottom.y0)
    
        fig.text(
            RUN_LABEL_X,
            y_center,
            f"Run {run_letter}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=FONT_SIZE
        )
    
    # ----------------------------------
    # Save/show
    # ----------------------------------
    if SAVE_PLOTS:
        fig.savefig(
            PLOTS_FOLDER_DIR / "group_bandplots_and_optional_pbc.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()
    
    
    print()
    print(f"Finished generating group results: Group {group_num}")
    print("=================================================")
    print()
    print()




if __name__ == "__main__":
    pass




###### Old plotting code ######


## Plot true rates (Example A) ##

# if PLOT_EXTRAS and not SAVE_PLOTS:
#     plt.figure(figsize=(TEXTWIDTH_IN, 4))
#     for col in true_lams.columns:
#         plt.plot(K_SERIES, true_lams[col], label=col)
#     plt.xlabel("k")
#     plt.ylabel("Value")
#     plt.title("True rates over time")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     plt.close("all")



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

