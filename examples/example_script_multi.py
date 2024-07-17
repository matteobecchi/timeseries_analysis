"""
Example script for running onion_multi
"""

import matplotlib.pyplot as plt
import numpy as np
from example_plots import (
    plot_medoids_multi,
    plot_one_trj_multi,
    plot_output_multi,
    plot_pop_fractions,
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
plot_state_populations("Fig4.png", n_windows, labels)
plot_sankey("Fig5.png", labels, n_windows, [100, 200, 300, 400])

### Clustering at all the possible time resolution ###
TAU_WINDOWS_LIST = np.geomspace(3, 10000, 20, dtype=int)
BINS = 25
input_data = np.load(PATH_TO_INPUT_DATA)

n_dims = input_data.shape[0]
n_particles = input_data.shape[1]
n_frames = input_data.shape[2]

tra = np.zeros((len(TAU_WINDOWS_LIST), 3))
pop_list = []

for i, tau_window in enumerate(TAU_WINDOWS_LIST):
    n_windows = int(n_frames / tau_window)
    excess_frames = n_frames - n_windows * tau_window

    if excess_frames > 0:
        reshaped_data = np.reshape(
            input_data[:, :, :-excess_frames], (n_particles * n_windows, -1))
    else:
        reshaped_data = np.reshape(input_data, (n_particles * n_windows, -1))

    state_list, labels = onion_multi(reshaped_data, bins=BINS)

    list_pop = [state.perc for state in state_list]
    list_pop.insert(0, 1 - np.sum(np.array(list_pop)))

    tra[i][0] = tau_window
    tra[i][1] = len(state_list)
    tra[i][2] = list_pop[0]
    pop_list.append(list_pop)

plot_time_res_analysis("Fig6.png", tra)
plot_pop_fractions("Fig7.png", pop_list)

plt.show()
