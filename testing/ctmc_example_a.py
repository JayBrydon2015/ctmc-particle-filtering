# -*- coding: utf-8 -*-

"""

Bootstrap PF for inferring rates of the CTMC in Example A.

"""


###### CONSTANTS ######

SIMULATE_DATA_MANUALLY = True # Simulate data manually (when J == 1 & K == 300)

EXAMPLE_FOLDER_NAME   = "CTMC_ExampleA_Figs"


###### IMPORTS ######

import numpy as np

from ctmc_modules.ctmc_ssms import (
    get_gamma_params_from_mean_var,
    CTMC,
    GP_CTMC
)

from rates_simulation.true_rates_simulation_funtions import (
    simulate_example_a, simulate_data, simulate_data_manually_example_a
)

from testing.run_results_generation import (
    generate_run_results_example_a
)


# %%

#### CTMC SSM & Other Parameters #####

n = 3

mu0  = np.array([1, 1, 1, 1, 5, 1]) # For P(lams_0), a vague prior
var0 = np.array([4, 4, 4, 4, 6, 4]) # For P(lams_0), a vague prior

num_runs = 9
runs_table_dictionary = {
    "N":   [5000, 5000, 5000, 5000, 100, 5000, 5000, 5000, 5000],
    # "N":   [50000, 50000, 50000, 50000, 100,   50000, 50000, 50000, 50000],
    "dt":  [0.01,  0.01,  0.01,  0.01,  0.01,  1,     0.002, 0.01,  0.01 ],
    "K":   [300,   300,   300,   900,   300,   300,   900,   300,   300  ],
    "J":   [1,     8,     10000, 1000,  1000,  1000,  1000,  1000,  1000 ],
    "C":   [1,     1,     1,     1,     1,     1,     1,     0.5,   2    ],
    "TV":  [False, False, False, False, False, False, False, True,  True ],
    "pbc": [False, False, False, True,  False, False, True,  False, False],
}
assert all(len(row) == num_runs for row in runs_table_dictionary.values())

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0, 0, 0, 1, 0]
bp_ymax = [3, 2, 2, 3, 9, 2]
y_ticks = [
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2, 3],
    [2, 4, 6, 8],
    [0, 1, 2],
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
    
    y_init = np.array([0 for _ in range(J)]) # RWs initial config y_{-1}
    
    a0, b0 = get_gamma_params_from_mean_var(mu0, var0)
    
    ## Create the SSM object ##
    
    ctmc_ssm = CTMC(
        n = n,
        J = J,
        delta_t = delta_t,
        C = C,
        a0 = a0,
        b0 = b0,
        y_init = y_init,
        TV = TV,
        px_verbose = True
    )
    
    ## Simulate true rates (true_states) and state vectors (data) ##
    
    true_states = simulate_example_a(K=K)
    
    if SIMULATE_DATA_MANUALLY and J == 1 and K == 300:
        
        data = simulate_data_manually_example_a()
        
    else:
        
        data = simulate_data(
            true_rates=true_states,
            n=n,
            J=J,
            delta_t=delta_t,
            y_init=y_init
        )
    
    ## Generate the results of the run ##
    
    generate_run_results_example_a(
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
    )


# %%

#### GP CTMC SSM & Other Parameters #####

mu0    = np.array([1, 1, 1, 1, 5, 1]) # For P(llams0), a vague prior
scale0 = np.array([1, 1, 1, 1, 1, 1]) # For P(llams0), a vague prior

# By convention, GP runs come after non-GP runs.
num_gp_runs = 5
runs_table_dictionary = {
    "N":   [5000,  5000,  5000,  5000,  5000 ],
    #"N":   [50000, 50000, 50000, 50000, 50000],
    "dt":  [0.01,  0.01,  0.01,  0.01,  0.01 ],
    "K":   [300,   300,   600,   600,   600  ],
    "J":   [1,     8,     10000, 1000,  1000 ],
    "l":   [0.8,   0.8,   0.8,   1.5,   0.8  ],
    "C":   [1,     1,     1,     1,     2    ],
    "pbc": [False, False, False, False, False],
}
assert all(len(row) == num_gp_runs for row in runs_table_dictionary.values())

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0, 0, 0, 2, 0]
bp_ymax = [3, 2, 2, 3, 10, 2]
y_ticks = [
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2, 3],
    [2, 4, 6, 8, 10],
    [0, 1, 2],
]


#### GENERATE RESULTS FOR EACH RUN: GP RUNS ####

for i in range(num_gp_runs): # Later GP runs
    
    ## Parameters ##
    
    N = runs_table_dictionary["N"][i]
    delta_t = runs_table_dictionary["dt"][i]
    K = runs_table_dictionary["K"][i]
    J = runs_table_dictionary["J"][i]
    l = runs_table_dictionary["l"][i]
    C = runs_table_dictionary["C"][i]
    
    y_init = np.array([0 for _ in range(J)]) # RWs initial config y_{-1}
    
    ## Create the SSM object ##
    
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
    
    
    ## Simulate true rates (true_states) and state vectors (data) ##
    
    true_states = simulate_example_a(K=K)
    
    if SIMULATE_DATA_MANUALLY and J == 1 and K == 300:
        
        data = simulate_data_manually_example_a()
        
    else:
        
        data = simulate_data(
            true_rates=true_states,
            n=n,
            J=J,
            delta_t=delta_t,
            y_init=y_init
        )
    
    ## Generate the results of the run ##
    
    generate_run_results_example_a(
        i=num_runs + i,
        ctmc_ssm=ctmc_ssm,
        true_states=true_states,
        data=data,
        example_folder_name=EXAMPLE_FOLDER_NAME,
        N=N,
        K=K,
        n=n,
        J=J,
        exp_X=True,
        bp_ymin=bp_ymin,
        bp_ymax=bp_ymax,
        y_ticks=y_ticks,
    )

