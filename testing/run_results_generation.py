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

from ctmc_modules.ctmc_ssms import lams_idx_to_gen_pos


###### CONSTANTS ######

SAVE_PLOTS  = True  # Save plots to a folder (True) or show them (False)
PLOT_EXTRAS = False # Plot extra stuff (True rates, KDEs and PW scatter plots)

PLOTS_ROOT_FOLDER_DIR = Path(__file__).parent / "generated_plots"
PLOTS_ROOT_FOLDER_DIR.mkdir(exist_ok=True)


###### PLOTTING LIBRARY IMPORTS & PLOTTING CONSTANTS ######

if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg') # Must be called before importing pyplot
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter

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


def make_true_lams_dataframe(*, true_states, n, k_series):
    """ Creates a pandas data frame storing the true rates. """
    
    lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                          for i in range(true_states[0].shape[1])]
    
    true_lams = pd.DataFrame(
        np.stack([true_state.reshape(-1) for true_state in true_states]),
        columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
        index=k_series
    ).rename_axis('k')
    
    return true_lams


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
            dims=("k"),
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


def use_compact_power_ticks(ax, nbins=3, font_size=12):
    """
    Format y-axis ticks using a power-of-ten multiplier.

    Example:
        0.00, 0.01, 0.02 -> 0, 1, 2 with 1e-2 above axis.
    """

    ymin, ymax = ax.get_ylim()
    max_abs = max(abs(ymin), abs(ymax))

    if max_abs == 0:
        return

    # Choose exponent. For values around 0.01, exponent=-2.
    # For values around 1, exponent=0.
    exponent = int(np.floor(np.log10(max_abs)))
    scale = 10.0 ** exponent

    # If exponent is 0, no scaling is useful.
    show_power_label = exponent != 0

    scaled_ymin = ymin / scale
    scaled_ymax = ymax / scale

    # Do NOT force integer=True here, because it can produce bad ticks
    # when the scaled range is narrow, e.g. 0.7 to 1.1.
    locator = MaxNLocator(nbins=nbins, min_n_ticks=2)
    scaled_ticks = locator.tick_values(scaled_ymin, scaled_ymax)

    scaled_ticks = scaled_ticks[
        (scaled_ticks >= scaled_ymin) & (scaled_ticks <= scaled_ymax)
    ]

    ax.set_yticks(scaled_ticks * scale)

    def tick_format(y, _):
        val = y / scale

        # Integer-looking labels when possible
        if abs(val - round(val)) < 1e-8:
            return f"{int(round(val))}"

        # Otherwise allow one decimal to avoid 1, 1, 1 bugs
        return f"{val:.1f}".rstrip("0").rstrip(".")

    if show_power_label:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    else:
        # No 1e0 label; just show normal values.
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:g}")
        )

    # Remove old custom power labels
    for text in list(ax.texts):
        if getattr(text, "_is_power_tick_label", False):
            text.remove()

    if show_power_label:
        offset_text = ax.text(
            0.0,
            1.02,
            f"1e{exponent}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=font_size,
        )
        offset_text._is_power_tick_label = True


def use_strict_integer_power_ticks(ax, max_ticks=4, font_size=12):
    """
    Force y-axis tick labels to be integers after scaling by a power of 10.

    Examples:
        0.00, 0.01, 0.02 -> 0, 1, 2 with 1e-2 above axis
        1, 2, 3          -> 1, 2, 3 with no 1e0
    """

    ymin, ymax = ax.get_ylim()

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        return

    max_abs = max(abs(ymin), abs(ymax))

    if max_abs == 0:
        return

    base_exp = int(np.floor(np.log10(max_abs)))

    chosen_exp = None
    chosen_ticks = None

    # Try a range of powers of 10.
    # Larger exp => coarser scaled units.
    # Smaller exp => finer scaled units.
    for exp in range(base_exp + 3, base_exp - 10, -1):
        scale = 10.0 ** exp

        scaled_ymin = ymin / scale
        scaled_ymax = ymax / scale

        tick_start = int(np.ceil(scaled_ymin))
        tick_end = int(np.floor(scaled_ymax))

        n_ticks = tick_end - tick_start + 1

        # Important: check n_ticks BEFORE np.arange
        if 2 <= n_ticks <= max_ticks:
            chosen_exp = exp
            chosen_ticks = np.arange(tick_start, tick_end + 1)
            break

    # Fallback: use a coarser exponent so we don't make billions of ticks
    if chosen_exp is None:
        for exp in range(base_exp + 6, base_exp - 10, -1):
            scale = 10.0 ** exp

            scaled_ymin = ymin / scale
            scaled_ymax = ymax / scale

            tick_start = int(np.ceil(scaled_ymin))
            tick_end = int(np.floor(scaled_ymax))

            n_ticks = tick_end - tick_start + 1

            if 1 <= n_ticks <= max_ticks:
                chosen_exp = exp
                chosen_ticks = np.arange(tick_start, tick_end + 1)
                break

    if chosen_exp is None or chosen_ticks is None or len(chosen_ticks) == 0:
        return

    scale = 10.0 ** chosen_exp
    ax.set_yticks(chosen_ticks * scale)

    if chosen_exp == 0:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{int(round(y))}")
        )
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{int(round(y / scale))}")
        )

    # Remove old custom exponent labels
    for text in list(ax.texts):
        if getattr(text, "_is_power_tick_label", False):
            text.remove()

    # Suppress unhelpful 1e0
    if chosen_exp != 0:
        offset_text = ax.text(
            0.0,
            1.02,
            f"1e{chosen_exp}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=font_size,
        )
        offset_text._is_power_tick_label = True


###### EXAMPLE A RUN RESULTS GENERATION ######


def generate_run_results_example_a(*,
        i,
        ctmc_ssm,
        true_states,
        data,
        example_folder_name,
        N, K, n, J,
        plot_bin_counts=False,
        exp_X=False,
        bp_ymin=None,
        bp_ymax=None,
        y_ticks=None
    ):
    
    """
    Generate the results of a run of Example A:
    - the data plot,
    - the ESS plot,
    - the band plots, and
    - any extras if PLOT_EXTRAS is True and SAVE_PLOTS is False.
    
    Inputs
    ------
    exp_X: True if log-rates are modelled in the SSM (this is the case with
      the Gaussian process SSM); False if not. If True, log-rates need to
      be exponentiated.
    """
    
    k_series = np.arange(K + 1) # [0, 1, ..., K]
    # TIME_POINTS = delta_t * K_SERIES # [t_0, t_1, ..., t_K], given t_0 = 0
    
    run_letter = map_num_to_letter(i)
    PLOTS_FOLDER_DIR = (
        PLOTS_ROOT_FOLDER_DIR / 
        f"{example_folder_name}/Run_{run_letter}"
    )
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating run results: Run {run_letter}")
    print("=================================================")
    print()
    
    
    ###### Store true rates in Pandas dataframe ######
    
    true_lams_df = make_true_lams_dataframe(
        true_states = true_states,
        n = n,
        k_series = k_series
    )
    
    
    ###### Run particle filter ######
    
    ds_boot = run_particle_filter(
        ctmc_ssm = ctmc_ssm,
        data = data,
        N = N,
        true_lams_df = true_lams_df, 
        k_series = k_series,
        exp_X = exp_X,
    )
    
    
    ###### DATA, ESS, & BAND PLOTS ######
    
    if J == 1: # Stack all of them vertically
    
        ## Initialise subplots ##
    
        fig, axes = plt.subplots(
            6, 1,
            sharex=True,
            figsize=(TEXTWIDTH_IN, 9.5),
            gridspec_kw={
                "height_ratios": [0.7, 1.0] + [1.8] * 4
            }
        )
        
        rw_ax = axes[0]
        ess_ax = axes[1]
        band_axes = axes[2:]
        
        ## Data plot ##
        
        data_plot = np.vstack(data) + 1
        
        rw_ax.plot(k_series, data_plot[:, 0], lw=2)
        
        rw_ax.set_ylim(0.8, 3.2)
        rw_ax.set_ylabel("State")
        rw_ax.grid(True)
        
        ## ESS plot ##
        
        ess_ax.plot(
            k_series,
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
            
            latex_symbol, rate_name_img = get_latex_rate_symbol_ex_a(p, q)
            
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
                k_series,
                true_lams_df[lam].values,
                color="blue",
                lw=2,
                label="True rate"
            )
            
            ax.plot(
                k_series,
                median,
                color="orange",
                lw=2,
                label="PF median"
            )
            
            ax.fill_between(
                k_series,
                lq,
                uq,
                color="orange",
                alpha=0.3
            )
            
            ax.set_title(latex_symbol, pad=2)
            ax.set_ylabel("Value")
            ax.grid(True)
            
            if band_ax_idx == 0:
                ax.legend()
            
            band_ax_idx += 1
            
        ## Formatting options ##
            
        axes[-1].set_xlabel("k")
    
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.10,
            top=0.94,
            hspace=0.3,
            wspace=0.28
        )
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        # fig.tight_layout()
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "J1_all_plots_combined.pdf",
                format="pdf",
                bbox_inches='tight',
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    else: # Plot band plots separately; ESS and data plotted together (J == 8)
        
        if J == 8:
            
            data_plot = np.vstack(data)
            data_plot += 1 # '0' -> State 1, etc
            
            fig = plt.figure(
                figsize=(TEXTWIDTH_IN, 3)
            )
            
            gs = fig.add_gridspec(
                nrows=4,
                ncols=3,
                width_ratios=[1, 1, 1.15],
                hspace=0.12,
                wspace=0.2
            )
            
            axes = []
            for row in range(4):
                for col in range(2):
                    share_ax = axes[0] if axes else None
                    axes.append(
                        fig.add_subplot(gs[row, col], sharex=share_ax)
                    )
            
            ess_ax = fig.add_subplot(gs[:, 2])
            
            for j in range(J):
                axes[j].plot(k_series, data_plot[:, j], lw=2)
                axes[j].grid(True)
                axes[j].set_ylim(0.8, 3.2)
                axes[j].set_xlim(-10, 310)
            
            for ax in axes[:-2]:
                ax.tick_params(axis="x", labelbottom=False)
            
            for ax in axes[-2:]:
                ax.set_xlabel("k")
                
            for ax in axes:
                ax.set_xticks([0, K / 3, 2 * K / 3, K])
            
            ess_ax.plot(k_series, ds_boot['ESS'], color="red", lw=1)
            ess_ax.set_xlabel("k")
            ess_ax.set_xlim(-10, 310)
            ess_ax.set_xticks([0, K / 3, 2 * K / 3, K])
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
            # fig.tight_layout()
            
            if SAVE_PLOTS:
                fig.savefig(
                    PLOTS_FOLDER_DIR / "rw_data_and_ess.pdf",
                    format="pdf",
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()
        
        
        ## BAND PLOTS ##
        
        fig, axes = plt.subplots(
            2, 3,
            figsize=(TEXTWIDTH_IN, 3),
            sharex=True
        )
        axes = axes.flatten()  # makes indexing easy: 0..5
        
        for lam_idx, lam in enumerate(true_lams_df.columns):
            ax = axes[lam_idx]
    
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            latex_symbol, rate_name_img = get_latex_rate_symbol_ex_a(p, q)
            
            median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
            lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
            uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    
            ax.plot(
                k_series,
                true_lams_df[lam].values,
                label="True rate",
                color='blue', alpha=0.8, lw=1
            )
    
            ax.plot(
                k_series,
                median,
                label="PF median",
                color="orange", alpha=0.7, lw=1
            )
    
            ax.fill_between(
                k_series,
                y1=lq,
                y2=uq,
                color="orange", alpha=0.3, lw=1
            )
            
            ax.set_title(f"{latex_symbol}", pad=2)
            if lam_idx == 2:
                ax.legend(loc=1, framealpha=0.5)
        
        for ax in axes:
            ax.set_xticks([0, int(K/3), int(2*K/3), K])
        
        for ax in axes[3:]:
            ax.set_xlabel("k")
        
        if bp_ymin is not None and bp_ymax is not None:
            for i, ax in enumerate(axes):
                ax.set_ylim(
                    bottom=bp_ymin[i] - 0.1,
                    top=bp_ymax[i] + 0.1
                )
        
        if y_ticks is not None:
            for i, ax in enumerate(axes):
                ax.set_yticks(y_ticks[i])
        
        plt.subplots_adjust(hspace=0.28)
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        
        # for ax in axes:
        #     use_strict_integer_power_ticks(
        #         ax,
        #         max_ticks=4,
        #         font_size=LATEX_FONT_SIZE
        #     )
        
        # fig.tight_layout()
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "all_bandplots_quantiles.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    ###### BIN COUNTS OVER TIME PLOT ######
    
    if plot_bin_counts:
        
        counts = np.array([
            np.bincount(data_k.reshape(-1), minlength=n)
            for data_k in data
        ])
        
        proportions = (
            counts /
            np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
        )
        
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3))
        
        for value in range(n):
            ax.plot(
                k_series,
                proportions[:, value],
                # Add 1 to label because data is between 0 and n-1 originally
                label=f"{value + 1}",
                lw=2, alpha=0.6
            )
        
        ax.set_xlabel("k")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend(title="State")
        ax.set_xticks([0, int(K/3), int(2*K/3), K])
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        fig.tight_layout()
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "state_bin_counts.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    print()
    print(f"Finished generating run results: Run {run_letter}")
    print("=================================================")
    print()
    print()


###### EXAMPLE B RUN RESULTS GENERATION ######


def generate_run_results_example_b(*,
        i,
        ctmc_ssm,
        true_states,
        data,
        example_folder_name,
        N, K, n, J,
        plot_bin_counts=False,
        exp_X=False,
        bp_ymin=None,
        bp_ymax=None,
        y_ticks=None,
        generate_true_rates_plot=False,
    ):
    
    """
    Generate the results of a run of Example B:
    - the data plot;
    - the ESS plot;
    - the band plots.
    
    Inputs
    ------
    exp_X: True if log-rates are modelled in the SSM (this is the case with
      the Gaussian process SSM); False if not. If True, log-rates need to
      be exponentiated.
    """
    
    n = 2
    k_series = np.arange(K + 1) # [0, 1, ..., K]
    
    run_letter = map_num_to_letter(i)
    PLOTS_FOLDER_DIR = (
        PLOTS_ROOT_FOLDER_DIR / 
        f"{example_folder_name}/Run_{run_letter}"
    )
    PLOTS_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating run results: Run {run_letter}")
    print("=================================================")
    print()
    
    
    ###### Store true rates in Pandas dataframe ######
    
    true_lams_df = make_true_lams_dataframe(
        true_states = true_states,
        n = n,
        k_series = k_series
    )
    
    
    ###### Run particle filter ######
    
    ds_boot = run_particle_filter(
        ctmc_ssm = ctmc_ssm,
        data = data,
        N = N,
        true_lams_df = true_lams_df, 
        k_series = k_series,
        exp_X = exp_X,
    )
    
    
    ###### Generate plots ######
    
    ## TRUE LAMS/RATES PLOT ##
    
    if generate_true_rates_plot:
        
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
                k_series,
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
    
    
    ## BAND PLOTS ##
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH_IN, 2.5),
        sharex=True
    )
    axes = axes.flatten() # makes indexing easy
    
    for lam_idx, lam in enumerate(true_lams_df.columns):
        ax = axes[lam_idx]

        p, q = lams_idx_to_gen_pos(lam_idx, n)
        latex_symbol, _ = get_latex_rate_symbol(p, q)
        
        median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
        lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
        uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)

        ax.plot(
            k_series,
            true_lams_df[lam].values,
            label="True rate",
            color='blue', alpha=0.8, lw=1
        )

        ax.plot(
            k_series,
            median,
            label="PF median",
            color="orange", alpha=0.7, lw=1
        )

        ax.fill_between(
            k_series,
            y1=lq,
            y2=uq,
            color="orange", alpha=0.3, lw=1
        )
        
        ax.set_title(f"{latex_symbol}", pad=2)
        if lam_idx == 0:
            ax.legend(loc="upper left", framealpha=0.5)
    
    for ax in axes:
        ax.set_xticks([0, int(K/3), int(2*K/3), K])
    
    for ax in axes:
        ax.set_xlabel("k")
    
    if bp_ymin is not None and bp_ymax is not None:
        for i, ax in enumerate(axes):
            ax.set_ylim(
                bottom=bp_ymin[i] - 0.1,
                top=bp_ymax[i] + 0.1
            )
    
    if y_ticks is not None:
        for i, ax in enumerate(axes):
            ax.set_yticks(y_ticks[i])
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        fig.savefig(
            PLOTS_FOLDER_DIR / "all_bandplots_quantiles.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()
    
    
    ###### BIN COUNTS OVER TIME PLOT ######
    
    if plot_bin_counts:
        
        counts = np.array([
            np.bincount(data_k.reshape(-1), minlength=n)
            for data_k in data
        ])
        
        proportions = (
            counts /
            np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
        )
        
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3))
        
        for value in range(n):
            ax.plot(
                k_series,
                proportions[:, value],
                # Add 1 to label because data is between 0 and n-1 originally
                label=f"{value + 1}",
                lw=2, alpha=0.6
            )
        
        ax.set_xlabel("k")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend(title="State")
        ax.set_xticks([0, int(K/3), int(2*K/3), K])
        
        apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
        
        if SAVE_PLOTS:
            fig.savefig(
                PLOTS_FOLDER_DIR / "state_bin_counts.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    
    print()
    print(f"Finished generating run results: Run {run_letter}")
    print("=================================================")
    print()
    print()


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

