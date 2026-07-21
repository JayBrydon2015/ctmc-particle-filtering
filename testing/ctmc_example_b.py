# -*- coding: utf-8 -*-

"""

Bootstrap PF for inferring rates of the CTMC in Example B.

"""


###### CONSTANTS ######

SIMULATE_DATA_MANUALLY = True # Simulate data manually (when J == 1 & K == 300)

EXAMPLE_FOLDER_NAME   = "CTMC_ExampleB_Figs"


###### IMPORTS ######

import numpy as np

from ctmc_modules.ctmc_ssms import (
    get_gamma_params_from_mean_var,
    CTMC,
    GP_CTMC
)

from rates_simulation.true_rates_simulation_funtions import (
    simulate_sine_rates_n2, simulate_data
)

from testing.run_results_generation import (
    generate_run_results_example_b
)


# %%

## CTMC SSM Parameters ##

n = 2

mu0  = np.array([1.8, 3.8])
var0 = np.array([2,   2  ])

a0, b0 = get_gamma_params_from_mean_var(mu0, var0)

num_runs = 5
runs_table_dictionary = {
    "N":   [5000,  5000,  5000,  5000,  5000 ],
    #"N":  [50000, 50000, 50000, 50000, 50000],
    "dt":  [0.01,  0.01,  0.01,  0.01,  0.005],
    "K":   [300,   300,   300,   300,   300  ],
    "J":   [10,    100,   1000,  1000,  1000 ],
    "C":   [2,     2,     2,     1,     1    ],
    "TV":  [True,  True,  True,  False, False],
    "pbc": [False, False, True,  False, False],
}
assert all(len(row) == num_runs for row in runs_table_dictionary.values())

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0]
bp_ymax = [6, 6]
y_ticks = [
    [0, 2, 4, 6],
    [0, 2, 4, 6],
]


#### GENERATE RESULTS FOR EACH RUN: NON-GP RUNS ####

for i in range(num_runs): # Run A, B, C, etc

    ## Parameters ##
    
    N = runs_table_dictionary["N"][i]
    delta_t = runs_table_dictionary["dt"][i]
    K = runs_table_dictionary["K"][i]
    J = runs_table_dictionary["J"][i]
    TV = runs_table_dictionary["TV"][i]
    C = runs_table_dictionary["C"][i]
    plot_bin_counts = runs_table_dictionary["pbc"][i]
    
    ## Create the SSM object ##
    
    ctmc_ssm = CTMC(
        n = n,
        J = J,
        delta_t = delta_t,
        C = C,
        a0 = a0,
        b0 = b0,
        y_init = None,
        TV = TV,
        px_verbose = True
    )
    
    y_init = ctmc_ssm.y_init
    
    ## Simulate true rates (true_states) and state vectors (data) ##
    
    true_states = simulate_sine_rates_n2(K=K)
        
    data = simulate_data(
        true_rates=true_states,
        n=n,
        J=J,
        delta_t=delta_t,
        y_init=y_init
    )
    
    ## Generate the results of the run ##
    
    generate_run_results_example_b(
        i=i,
        ctmc_ssm=ctmc_ssm,
        true_states=true_states,
        data=data,
        example_folder_name=EXAMPLE_FOLDER_NAME,
        N=N,
        K=K,
        n=n,
        J=J,
        plot_bin_counts=plot_bin_counts,
        exp_X=False,
        bp_ymin=bp_ymin,
        bp_ymax=bp_ymax,
        y_ticks=y_ticks,
        generate_true_rates_plot=True,
    )


# %%

## GP CTMC SSM Parameters ##

n = 2

mu0  = np.array([1.8, 3.8])
scale0 = np.array([1, 1])

a0, b0 = get_gamma_params_from_mean_var(mu0, var0)

# By convention, GP runs come after non-GP runs.
num_gp_runs = 7
runs_table_dictionary = {
    "N":   [5000,  5000,  5000,  5000,  5000,  5000,  5000 ],
    #"N":   [50000, 50000, 50000, 50000, 50000, 50000, 50000],
    "dt":  [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01 ],
    "K":   [300,   300,   300,   300,   300,   300,   300  ],
    "J":   [10,    100,   1000,  1000,  1000,  1000,  1000 ],
    "l":   [0.8,   0.8,   0.8,   0.8,   0.5,   1.5,   2    ],
    "C":   [6,     6,     6,     12,    6,     6,     6    ],
    "pbc": [False, False, False, False, False, False, False],
}
assert all(len(row) == num_gp_runs for row in runs_table_dictionary.values())

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0]
bp_ymax = [6, 6]
y_ticks = [
    [0, 2, 4, 6],
    [0, 2, 4, 6],
]


#### GENERATE RESULTS FOR EACH RUN: GP RUNS ####

for i in range(num_gp_runs-3, num_gp_runs): # Run A, B, C, etc

    ## Parameters ##
    
    N = runs_table_dictionary["N"][i]
    delta_t = runs_table_dictionary["dt"][i]
    K = runs_table_dictionary["K"][i]
    J = runs_table_dictionary["J"][i]
    l = runs_table_dictionary["l"][i]
    C = runs_table_dictionary["C"][i]
    plot_bin_counts = runs_table_dictionary["pbc"][i]
    
    ## Create the SSM object ##
    
    ctmc_ssm = GP_CTMC(
        n = n,
        J = J,
        delta_t = delta_t,
        l = l,
        C = C,
        mu0 = mu0,
        scale0 = scale0,
        y_init = None,
        px_verbose = True
    )
    
    y_init = ctmc_ssm.y_init
    
    ## Simulate true rates (true_states) and state vectors (data) ##
    
    true_states = simulate_sine_rates_n2(K=K)
        
    data = simulate_data(
        true_rates=true_states,
        n=n,
        J=J,
        delta_t=delta_t,
        y_init=y_init
    )
    
    ## Generate the results of the run ##
    
    generate_run_results_example_b(
        i=num_runs + i,
        ctmc_ssm=ctmc_ssm,
        true_states=true_states,
        data=data,
        example_folder_name=EXAMPLE_FOLDER_NAME,
        N=N,
        K=K,
        n=n,
        J=J,
        plot_bin_counts=plot_bin_counts,
        exp_X=True,
        bp_ymin=bp_ymin,
        bp_ymax=bp_ymax,
        y_ticks=y_ticks,
        generate_true_rates_plot=False,
    )

