# -*- coding: utf-8 -*-

"""

Functionality that generates the results of a run for any of the CTMC case
study examples.

"""


###### CONSTANTS ######

SAVE_PLOTS  = True  # Save plots to a folder (True) or show them (False)
PLOT_EXTRAS = False # Plot extra stuff (True rates, KDEs and PW scatter plots)

PLOT_ROOT_FOLDER_NAME = "generated_plots"
TEXTWIDTH_IN = 6.614


###### IMPORTS ######

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg') # Must be called before importing pyplot
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter
import seaborn as sns

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments

from ctmc_modules.ctmc_ssms import lams_idx_to_gen_pos


###### GLOBAL PLOT FONT SIZES ######

LATEX_FONT_SIZE = 12

ESS_FORMATTER = ScalarFormatter(useMathText=False)
ESS_FORMATTER.set_scientific(True)
ESS_FORMATTER.set_powerlimits((4, 4))


###### FUNCTIONS ######


def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate used in the image file
        name.
    """
    
    if p == 1 and q == 2:
        return "$\\varepsilon$", "Eps"
    elif p == 2 and q == 3:
        return "$\\delta$", "Delta"
    elif p == 3 and q == 1:
        return "$\\phi$", "Phi"
    else:
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


def apply_font_sizes(fig, size):
    for ax in fig.axes:
        ax.title.set_fontsize(size)
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        ax.tick_params(axis="both", labelsize=size - 2)

        ax.xaxis.get_offset_text().set_fontsize(size - 2)
        ax.yaxis.get_offset_text().set_fontsize(size - 2)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(size - 1)
            legend.get_title().set_fontsize(size - 1)


def use_compact_power_ticks(ax, nbins=3, font_size=12):
    """
    Format y-axis ticks as one-digit-ish integers with a 1eX multiplier above.

    Example:
        raw ticks: 0.00, 0.01, 0.02
        labels:    0,    1,    2
        offset:    1e-2
    """

    ymin, ymax = ax.get_ylim()
    max_abs = max(abs(ymin), abs(ymax))

    if max_abs == 0:
        return

    # Choose scale so largest visible value is roughly 1--9
    exponent = int(np.floor(np.log10(max_abs)))
    scale = 10.0 ** exponent

    # Work in scaled coordinates
    scaled_ymin = ymin / scale
    scaled_ymax = ymax / scale

    locator = MaxNLocator(nbins=nbins, integer=True, min_n_ticks=2)
    scaled_ticks = locator.tick_values(scaled_ymin, scaled_ymax)

    # Keep only ticks inside the actual y-limits
    scaled_ticks = scaled_ticks[
        (scaled_ticks >= scaled_ymin) & (scaled_ticks <= scaled_ymax)
    ]

    ax.set_yticks(scaled_ticks * scale)

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y / scale:.0f}")
    )

    # Remove any old custom offset labels if this function gets called repeatedly
    for text in ax.texts:
        if getattr(text, "_is_power_tick_label", False):
            text.remove()

    # Add the 1eX label above the y-axis
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


def generate_run_results(*,
        i,
        ctmc_ssm,
        true_states,
        data,
        example_folder_name,
        N, K, n, J,
        plot_bin_counts=False,
        exp_X=False
    ):
    
    """
    Generate the results of a run:
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
    
    K_SERIES = np.arange(K + 1)      # [0, 1, ..., K]
    # TIME_POINTS = delta_t * K_SERIES # [t_0, t_1, ..., t_K], given t_0 = 0
    
    run_letter = map_num_to_letter(i)
    FOLDER_PATH = Path(
        f"{PLOT_ROOT_FOLDER_NAME}/{example_folder_name}/Run_{run_letter}"
    )
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating run results: Run {run_letter}")
    print("=================================================")
    print()
    
    
    ###### Store true rates in Pandas dataframe ######
    
    lams_gen_positions = [lams_idx_to_gen_pos(i, n)
                          for i in range(true_states[0].shape[1])]
    
    true_lams = pd.DataFrame(
        np.stack([true_state.reshape(-1) for true_state in true_states]),
        columns=[f"λ_{p}{q}" for p, q in lams_gen_positions],
        index=K_SERIES
    ).rename_axis('k')
    
    
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
            np.stack([pf_boot.hist.X[k] for k in K_SERIES]),
            dims=("k", "particle", "lam"),
            coords={
                "k": K_SERIES,
                "lam": true_lams.columns.values,
            },
            name="Bootstrap PF Particles"
        ),
        
        'W': xr.DataArray(
            np.stack([pf_boot.hist.wgts[k].W for k in K_SERIES]),
            dims=("k", "weight"),
            coords={
                "k": K_SERIES
            },
            name="Bootstrap PF Weights"
        )
    })
    
    # If exp_X, exponentiate the log-rates
    if exp_X:
        ds_boot["X"] = np.exp(ds_boot["X"])
    
    
    ###### Calculate quantiles and add into ds_boot ######
    
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
        
        rw_ax.plot(K_SERIES, data_plot[:, 0], lw=2)
        
        rw_ax.set_ylim(0.8, 3.2)
        rw_ax.set_ylabel("State")
        rw_ax.grid(True)
        
        ## ESS plot ##
        
        ess_ax.plot(
            K_SERIES,
            pf_boot.summaries.ESSs,
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
        ess_ax.yaxis.get_offset_text().set_fontsize(LATEX_FONT_SIZE)
        ess_ax.grid(True)
        
        ## Band plots ##
        
        band_ax_idx = 0
        
        for lam_idx, lam in enumerate(true_lams.columns):
            
            if lam_idx in {1, 5}:
                continue
            
            ax = band_axes[band_ax_idx]
            
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            
            latex_symbol, rate_name_img = get_latex_rate_symbol(p, q)
            
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
                K_SERIES,
                true_lams[lam].values,
                color="blue",
                lw=2,
                label="True rate"
            )
            
            ax.plot(
                K_SERIES,
                median,
                color="orange",
                lw=2,
                label="PF median"
            )
            
            ax.fill_between(
                K_SERIES,
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
        
        apply_font_sizes(fig, LATEX_FONT_SIZE)
        # fig.tight_layout()
        
        if SAVE_PLOTS:
            fig.savefig(
                FOLDER_PATH / "J1_all_plots_combined.pdf",
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
                axes[j].plot(K_SERIES, data_plot[:, j], lw=2)
                axes[j].grid(True)
                axes[j].set_ylim(0.8, 3.2)
                axes[j].set_xlim(-10, 310)
            
            for ax in axes[:-2]:
                ax.tick_params(axis="x", labelbottom=False)
            
            for ax in axes[-2:]:
                ax.set_xlabel("k")
                
            for ax in axes:
                ax.set_xticks([0, 100, 200, 300])
            
            ess_ax.plot(K_SERIES, pf_boot.summaries.ESSs, color="red", lw=1)
            ess_ax.set_xlabel("k")
            ess_ax.set_xlim(-10, 310)
            ess_ax.set_xticks([0, 100, 200, 300])
            ess_ax.set_ylabel("ESS")
            ess_ax.yaxis.set_major_formatter(ESS_FORMATTER)
            ess_ax.yaxis.get_offset_text().set_fontsize(LATEX_FONT_SIZE)
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
            
            apply_font_sizes(fig, LATEX_FONT_SIZE)
            # fig.tight_layout()
            
            if SAVE_PLOTS:
                fig.savefig(
                    FOLDER_PATH / "rw_data_and_ess.pdf",
                    format="pdf",
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()
        
        
        ## BAND PLOTS ##
        
        # BP_MIN_YLIMS = [-0.1, -0.1, -0.1, -0.1, 0.5, -0.1]
        # BP_MAX_YLIMS = [2, 2, 2, 2, 7, 2]
        fig, axes = plt.subplots(
            2, 3,
            figsize=(TEXTWIDTH_IN, 3),
            sharex=True
        )
        axes = axes.flatten()  # makes indexing easy: 0..5
        
        for lam_idx, lam in enumerate(true_lams.columns):
            ax = axes[lam_idx]
    
            p, q = lams_idx_to_gen_pos(lam_idx, n)
            latex_symbol, rate_name_img = get_latex_rate_symbol(p, q)
            
            median = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.5)
            lq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.05)
            uq = ds_boot["X_quantiles"].sel(lam=lam, quantile=0.95)
    
            ax.plot(
                K_SERIES,
                true_lams[lam].values,
                label="True rate",
                color='blue', alpha=0.8, lw=1
            )
    
            ax.plot(
                K_SERIES,
                median,
                label="PF median",
                color="orange", alpha=0.7, lw=1
            )
    
            ax.fill_between(
                K_SERIES,
                y1=lq,
                y2=uq,
                color="orange", alpha=0.3, lw=1
            )
    
            # ax.set_ylim(BP_MIN_YLIMS[lam_idx], top=BP_MAX_YLIMS[lam_idx])
            ax.set_title(f"{latex_symbol}", pad=2)
            if lam_idx == 2:
                ax.legend(loc=1, framealpha=0.5)
        
        for ax in axes:
            ax.set_xticks([0, int(K/3), int(2*K/3), K])
        
        for ax in axes[3:]:
            ax.set_xlabel("k")
        
        plt.subplots_adjust(hspace=0.28)
        
        apply_font_sizes(fig, LATEX_FONT_SIZE)
        # fig.tight_layout()
        
        for ax in axes:
            use_compact_power_ticks(ax, nbins=3, font_size=LATEX_FONT_SIZE-2)
        
        if SAVE_PLOTS:
            fig.savefig(
                FOLDER_PATH / "all_bandplots_quantiles.pdf",
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
                K_SERIES,
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
        
        apply_font_sizes(fig, LATEX_FONT_SIZE)
        fig.tight_layout()
        
        if SAVE_PLOTS:
            fig.savefig(
                FOLDER_PATH / "state_bin_counts.pdf",
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

