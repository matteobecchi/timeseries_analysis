"""
Example script for running onion_clustering
"""

import matplotlib.pyplot as plt
import numpy as np
from example_plots import (
    plot_medoids_uni,
    plot_one_trj_uni,
    plot_output_uni,
    plot_pop_fractions,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)
from onion_clustering.onion_uni import onion_uni

#############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "onion_example_files/data/univariate_time-series.npy"
TAU_WINDOW = 5

### Optional parametrers ###
tmp_array = np.geomspace(2, 499, 50, dtype=int)
TAU_WINDOW_LIST = []
for item in tmp_array:
    if item not in TAU_WINDOW_LIST:
        TAU_WINDOW_LIST.append(item)
#############################################################################

input_data = np.load(PATH_TO_INPUT_DATA)[:, 1:]

state_list, labels, time_res_analysis, pop_list = onion_uni(
    input_data,
    tau_window=TAU_WINDOW,
    tau_window_list=TAU_WINDOW_LIST,
)

### These functions are examples of how to visualize the results
# plot_time_res_analysis("Fig1.png", time_res_analysis)
# plot_output_uni("Fig2.png", input_data, state_list)
plot_one_trj_uni("Fig3.png", 1234, input_data, labels)
plot_medoids_uni("Fig4.png", TAU_WINDOW, input_data, labels)
# plot_state_populations("Fig5.png", labels)
# plot_sankey("Fig6.png", labels, [100, 200, 300, 400])
# plot_pop_fractions("Fig7.png", pop_list)

plt.show()
