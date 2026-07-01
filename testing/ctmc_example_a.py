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
)

from rates_simulation.true_rates_simulation_funtions import (
    simulate_example_a, simulate_data, simulate_data_manually_example_a
)

from run_results_generation import (
    generate_run_results
)

# %%

#### CTMC SSM & Other Parameters #####

n = 3 # Number of states in the CTMC

mu0  = np.array([1, 1, 1, 1, 5, 1]) # For P(lams_0), a vague prior
var0 = np.array([4, 4, 4, 4, 6, 4]) # For P(lams_0), a vague prior

K = 300 # k = 0, ..., K

num_runs = 9
runs_table_dictionary = {
    "N": [50000] * num_runs,
    "delta_t": [0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005],
    "J": [1, 8, 1000, 10000, 1000, 1000, 1000, 1000, 1000],
    "TD": [False, False, False, False, True, False, False, True, True],
    "C": [1, 1, 1, 1, 1, 0.5, 2, 0.5, 2],
}
assert all(len(row) == num_runs for row in runs_table_dictionary.values())


#### GENERATE RESULTS FOR EACH RUN ####

for i in range(num_runs): # Run A, B, C, etc.
    
    ## Parameters ##
    
    N = runs_table_dictionary["N"][i]
    delta_t = runs_table_dictionary["delta_t"][i]
    J = runs_table_dictionary["J"][i]
    TD = runs_table_dictionary["TD"][i]
    C = runs_table_dictionary["C"][i]
    
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
        TD = TD,
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
    
    generate_run_results(
        i=i,
        ctmc_ssm=ctmc_ssm,
        true_states=true_states,
        data=data,
        example_folder_name=EXAMPLE_FOLDER_NAME,
        N=N,
        K=K,
        n=n,
        J=J,
    )

