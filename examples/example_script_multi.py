"""
Example script for running onion_multi
"""

import matplotlib.pyplot as plt
import numpy as np
from example_plots import (
    plot_medoids_multi,
    plot_one_trj_multi,
    plot_output_multi,
    plot_sankey,
    plot_state_populations,
)
from onion_clustering.onion_multi import onion_multi

##############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "onion_example_files/data/multivariate_time-series.npy"

### Clustering at one specific value of time resolution ###
TAU_WINDOW = 10
BINS = 25

input_data = np.load(PATH_TO_INPUT_DATA)

n_dims = input_data.shape[0]
n_particles = input_data.shape[1]
n_frames = input_data.shape[2]

n_windows = int(n_frames / TAU_WINDOW)
reshaped_data = np.reshape(input_data, (n_particles * n_windows, -1))
state_list, labels = onion_multi(reshaped_data, bins=BINS)

### These functions are examples of how to visualize the results
plot_output_multi("Fig1.png", input_data, state_list, labels, TAU_WINDOW)
plot_one_trj_multi("Fig2.png", 0, TAU_WINDOW, input_data, labels)
plot_medoids_multi("Fig3.png", TAU_WINDOW, reshaped_data, labels)
plot_state_populations("Fig4.png", labels)
plot_sankey("Fig5.png", labels, [100, 200, 300, 400])

### Clustering at all the possible time resolution ###

# plot_time_res_analysis("Fig6.png", time_res_analysis)
# plot_pop_fractions("Fig7.png", pop_list)

plt.show()
