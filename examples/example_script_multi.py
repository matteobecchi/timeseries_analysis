"""
Example script for running onion_clustering_2d
"""

import matplotlib.pyplot as plt
import numpy as np
from example_plots import (
    plot_medoids_multi,
    plot_one_trj_multi,
    plot_output_multi,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)
from onion_clustering.onion_multi import onion_multi

##############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "onion_example_files/data/multivariate_time-series.npy"
TAU_WINDOW = 10

### Optional parametrers ###
TAU_WINDOW_LIST = [5, 10, 15, 20]
BINS = 25
##############################################################################

input_data = np.load(PATH_TO_INPUT_DATA)

state_list, labels, time_res_analysis = onion_multi(
    input_data,
    tau_window=TAU_WINDOW,
    tau_window_list=TAU_WINDOW_LIST,
)

### These functions are examples of how to visualize the results
plot_time_res_analysis("Fig1.png", time_res_analysis)
plot_output_multi("Fig2.png", input_data, state_list, labels, TAU_WINDOW)
plot_one_trj_multi("Fig3.png", 0, TAU_WINDOW, input_data, labels)
plot_medoids_multi("Fig4.png", TAU_WINDOW, input_data, labels)
plot_state_populations("Fig5.png", labels)
plot_sankey("Fig6.png", labels, [100, 200, 300, 400])

plt.show()
