# -*- coding: utf-8 -*-

"""

Bootstrap PF for inferring rates of the CTMC in Case Study #4.

In this case study, the CMTC is high-dimensional as it has 5 states.

"""


###### CONSTANTS ######

N_L = 50000
t_K = 3 # What t_K should be
generate_true_rates_plot = True


###### IMPORTS ######

from testing.run_results_generation import (
    generate_group_plots_for_case_study,
    state_proportions_standalone_plot
)


# %%

#### CTMC SSM & Other Parameters ####

total_num_runs_nongp = 3
run_groups = [
    {
     "N":   [20   ],
     "dt":  [0.03 ],
     "K":   [100  ],
     "J":   [1000 ],
     "C":   [2    ],
     "TV":  ["CV" ],
     "pbc": [False],
    },
    
    
    {
     "N":   [N_L,   N_L  ],
     "dt":  [0.03,  0.03 ],
     "K":   [100,   100  ],
     "J":   [1000,  1000 ],
     "C":   [2,     0.40 ],
     "TV":  ["CV",  "CV" ],
     "pbc": [False, False],
    },
]
assert total_num_runs_nongp == sum(
    len(runs_dic["N"]) for runs_dic in run_groups
)
assert all(
    all(
        runs_dic["dt"][i] * runs_dic["K"][i] == t_K
        for i in range(len(runs_dic["dt"]))
    )
    for runs_dic in run_groups
)

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0, 0, 0, 0, 0]
bp_ymax = [5, 5, 5, 5, 5, 5]
y_ticks = [
    [0, 2, 4],
    [0, 2, 4],
    [0, 2, 4],
    [0, 2, 4],
    [0, 2, 4],
    [0, 2, 4],
]


#### GENERATE RESULTS FOR EACH GROUP/RUN: NON-GP RUNS ####

start_idx = 0

for i in range(len(run_groups)): # Group 1, 2, 3, etc
    
    runs_table_dictionary = run_groups[i]
    num_runs = len(runs_table_dictionary["N"])
    
    generate_group_plots_for_case_study(
        case_study_num = 4,
        group_num = i + 1,
        start_index = start_idx,
        runs_table_dictionary = runs_table_dictionary,
        gp = False,
        bp_ymin = bp_ymin,
        bp_ymax = bp_ymax,
        y_ticks = y_ticks,
        generate_true_rates_plot = generate_true_rates_plot
    )
    
    # Only need to generate true rates plot once
    if generate_true_rates_plot:
        generate_true_rates_plot = False
    
    start_idx += num_runs


# %%

#### Create standalone state proportions plot for 10,000 RWs ####

state_proportions_standalone_plot(
    case_study_num = 4,
    delta_t = 0.01,
    K = 300,
    J = 10000,
    t_K = 3,
    data = None, # Data created in function
)

