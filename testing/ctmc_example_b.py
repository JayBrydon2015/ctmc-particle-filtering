# -*- coding: utf-8 -*-

"""

Bootstrap PF for inferring rates of the CTMC in Example A.

"""


###### CONSTANTS ######

EXAMPLE_FOLDER_NAME   = "CTMC_ExampleB_Figs"

N_L = 5000


###### IMPORTS ######

from testing.run_results_generation import (
    generate_group_plots_example_b
)


# %%

#### CTMC SSM & Other Parameters ####

total_num_runs_nongp = 12
run_groups = [
    {
     "N":   [N_L,   N_L,   N_L,   N_L,   N_L  ],
     "dt":  [0.3,   0.1,   0.03,  0.01,  0.002],
     "K":   [10,    30,    100,   300,   1500 ],
     "J":   [1000,  1000,  1000,  1000,  1000 ],
     "C":   [2,     2,     2,     2,     2    ],
     "TV":  [True,  True,  True,  True,  True ],
     "pbc": [False, False, False, False, False],
    },
    
    {
     "N":   [N_L,   N_L,   N_L  ],
     "dt":  [0.03,  0.03,  0.03 ],
     "K":   [100,   100,   100  ],
     "J":   [10,    100,   10000],
     "C":   [2,     2,     2    ],
     "TV":  [True,  True,  True ],
     "pbc": [False, False, True ],
    },
    
    {
     "N":   [N_L,   N_L,   N_L,   N_L  ],
     "dt":  [0.03,  0.03,  0.03,  0.03 ],
     "K":   [100,   100,   100,   100  ],
     "J":   [1000,  1000,  1000,  1000 ],
     "C":   [0.2,   10,    1,     5    ],
     "TV":  [True,  True,  False, False],
     "pbc": [False, False, False, False],
    },
    
    
]
assert total_num_runs_nongp == sum(
    len(runs_dic["N"]) for runs_dic in run_groups
)

# Band plot ylim and ymax values, and y axis ticks
bp_ymin = [0, 0]
bp_ymax = [6, 6]
y_ticks = [
    [0, 2, 4, 6],
    [0, 2, 4, 6],
]


#### GENERATE RESULTS FOR EACH GROUP/RUN: NON-GP RUNS ####

start_idx = 0

for i in [2]:#range(len(run_groups)): # Group 1, 2, 3, etc
    
    runs_table_dictionary = run_groups[i]
    num_runs = len(runs_table_dictionary["N"])
    
    generate_group_plots_example_b(
        group_num = i + 1,
        start_index = start_idx,
        runs_table_dictionary = runs_table_dictionary,
        example_folder_name = EXAMPLE_FOLDER_NAME,
        gp = False, # True only if Gaussian process runs
        bp_ymin = bp_ymin,
        bp_ymax = bp_ymax,
        y_ticks = y_ticks
    )
    
    start_idx += num_runs


# %%

#### GENERATE TRUE RATES PLOT ####

generate_group_plots_example_b(
    group_num = i + 1,
    start_index = start_idx,
    runs_table_dictionary = runs_table_dictionary,
    example_folder_name = EXAMPLE_FOLDER_NAME,
    gp = False, # True only if Gaussian process runs
    bp_ymin = bp_ymin,
    bp_ymax = bp_ymax,
    y_ticks = y_ticks,
    generate_true_rates_plot=True
)


# %%

#### GP CTMC SSM & Other Parameters ####

# By convention, GP runs come after non-GP runs
total_num_runs_gp = 4
run_groups_gp = [
    {
     "N":   [N_L,   N_L  ],
     "dt":  [0.01,  0.01 ],
     "K":   [600,   600  ],
     "J":   [8,     10000],
     "l":   [0.8,   0.8  ],
     "C":   [1,     1    ],
     "pbc": [False, False],
    },
    
    {
     "N":   [N_L,   N_L  ],
     "dt":  [0.01,  0.01 ],
     "K":   [600,   600  ],
     "J":   [1000,  1000 ],
     "l":   [1.5,   0.8  ],
     "C":   [1,     2    ],
     "pbc": [False, False],
    },
]
assert total_num_runs_gp == sum(
    len(runs_dic["N"]) for runs_dic in run_groups_gp
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
    
    generate_group_plots_example_b(
        group_num = len(run_groups) + i + 1,
        start_index = start_idx,
        runs_table_dictionary = runs_table_dictionary,
        example_folder_name = EXAMPLE_FOLDER_NAME,
        gp = True, # True only if Gaussian process runs
        bp_ymin = bp_ymin,
        bp_ymax = bp_ymax,
        y_ticks = y_ticks
    )
    
    start_idx += num_runs

