"""
Example script for running onion_uni
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

### Clustering at one specific value of time resolution ###
TAU_WINDOW = 5

input_data = np.load(PATH_TO_INPUT_DATA)[:, 1:]
n_particles = input_data.shape[0]
n_frames = input_data.shape[1]

n_windows = int(n_frames / TAU_WINDOW)
frames_in_excess = n_frames - n_windows * TAU_WINDOW
reshaped_data = np.reshape(
    input_data[:, :-frames_in_excess], (n_particles * n_windows, TAU_WINDOW)
)

state_list, labels = onion_uni(reshaped_data)

### These functions are examples of how to visualize the results
plot_output_uni("Fig1.png", reshaped_data, n_windows, state_list)
plot_one_trj_uni("Fig2.png", 1234, reshaped_data, labels, n_windows)
plot_medoids_uni("Fig3.png", reshaped_data, labels)
plot_state_populations("Fig4.png", n_windows, labels)
plot_sankey("Fig5.png", labels, n_windows, [10, 20, 30, 40])

### Clustering at all the possible time resolution ###
TMP_LIST = np.geomspace(2, 499, num=20, dtype=int)
TAU_WINDOWS = [x for i, x in enumerate(TMP_LIST) if x not in TMP_LIST[:i]]

tra = np.zeros((len(TAU_WINDOWS), 3))
list_of_pop = []

for i, tau_window in enumerate(TAU_WINDOWS):
    n_windows = int(n_frames / tau_window)
    frames_in_excess = n_frames - n_windows * tau_window
    if frames_in_excess > 0:
        tmp_input_data = input_data[:, :-frames_in_excess]
    else:
        tmp_input_data = input_data
    reshaped_data = np.reshape(
        tmp_input_data,
        (n_particles * n_windows, tau_window),
    )

    state_list, labels = onion_uni(reshaped_data)

    pop_list = [state.perc for state in state_list]
    pop_list.insert(0, 1 - np.sum(np.array(pop_list)))
    list_of_pop.append(pop_list)

    tra[i][0] = tau_window
    tra[i][1] = len(state_list)
    tra[i][2] = pop_list[0]

### These functions are examples of how to visualize the results
plot_time_res_analysis("Fig6.png", tra)
plot_pop_fractions("Fig7.png", list_of_pop)

plt.show()
