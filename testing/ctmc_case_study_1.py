# -*- coding: utf-8 -*-

"""

Bootstrap PF for inferring rates of the CTMC in Case Study #1.

This case study models a faulty machine's state in continuous time with a
continuous-time Markov chain.

"""


###### CONSTANTS ######

N_L = 50000
t_K = 3 # What t_K should be

###### IMPORTS ######

from testing.run_results_generation import (
    generate_group_plots_for_case_study
)


# %%

#### CTMC SSM & Other Parameters ####

total_num_runs_nongp = 6
run_groups = [
    {
     "N":   [N_L  ],
     "dt":  [0.01 ],
     "K":   [300  ],
     "J":   [1    ],
     "C":   [1    ],
     "TV":  ["PV" ],
     "pbc": [False]
    },
    
    {
     "N":   [N_L,   N_L  ],
     "dt":  [0.01,  0.01 ],
     "K":   [300,   600  ],
     "J":   [8,     1000 ],
     "C":   [1,     1    ],
     "TV":  ["PV" , "PV" ],
     "pbc": [False, True ],
    },
    
    {
     "N":   [N_L,   N_L,   N_L  ],
     "dt":  [0.1,   0.01,  0.01 ],
     "K":   [30,    300,   300  ],
     "J":   [1000,  1000,  1000 ],
     "C":   [1,     2,     0.05 ],
     "TV":  ["PV" , "CV" , "CV" ],
     "pbc": [False, False, False]
    },
]
assert total_num_runs_nongp == sum(
    len(runs_dic["N"]) for runs_dic in run_groups
)
assert all(
    all(
        (runs_dic["dt"][i] * runs_dic["K"][i] == t_K or
        runs_dic["dt"][i] * runs_dic["K"][i] == 2 * t_K)
        for i in range(len(runs_dic["dt"]))
    )
    for runs_dic in run_groups
)


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


#### GENERATE RESULTS FOR EACH GROUP/RUN: NON-GP RUNS ####

start_idx = 0

for i in range(0, len(run_groups)): # Group 1, 2, 3, etc
    
    runs_table_dictionary = run_groups[i]
    num_runs = len(runs_table_dictionary["N"])
    
    generate_group_plots_for_case_study(
        case_study_num = 1,
        group_num = i + 1,
        start_index = start_idx,
        runs_table_dictionary = runs_table_dictionary,
        gp = False,
        bp_ymin = bp_ymin,
        bp_ymax = bp_ymax,
        y_ticks = y_ticks
    )
    
    start_idx += num_runs

# %%

#### GP CTMC SSM & Other Parameters ####

# By convention, GP runs come after non-GP runs
total_num_runs_gp = 3
run_groups_gp = [
    {
     "N":      [N_L,   N_L,   N_L  ],
     "dt":     [0.01,  0.01,  0.01 ],
     "K":      [300,   300,   300  ],
     "J":      [1000,  1000,  1000 ],
     "kappa":  [0.98,  0.4,   0.98 ],
     "sigma2": [0.04,  0.04,  1    ],
     "pbc":    [False, False, False],
    },
]
assert total_num_runs_gp == sum(
    len(runs_dic["N"]) for runs_dic in run_groups_gp
)
assert all(
    all(
        runs_dic["dt"][i] * runs_dic["K"][i] == t_K
        for i in range(len(runs_dic["dt"]))
    )
    for runs_dic in run_groups_gp
)

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

#### GENERATE RESULTS FOR EACH GROUP/RUN: GP RUNS ####

start_idx = total_num_runs_nongp

for i in range(len(run_groups_gp)): # Group 1, 2, 3, etc
    
    runs_table_dictionary = run_groups_gp[i]
    num_runs = len(runs_table_dictionary["N"])
    
    generate_group_plots_for_case_study(
        case_study_num = 1,
        group_num = len(run_groups) + i + 1,
        start_index = start_idx,
        runs_table_dictionary = runs_table_dictionary,
        gp = True, # True only if Gaussian process runs
        bp_ymin = bp_ymin,
        bp_ymax = bp_ymax,
        y_ticks = y_ticks
    )
    
    start_idx += num_runs

