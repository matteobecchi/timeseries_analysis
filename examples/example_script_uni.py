"""
Example script for running onion_clustering
"""

import matplotlib.pyplot as plt
import numpy as np
from example_plots import (
    plot_pop_fractions,
    plot_time_res_analysis,
)
from onion_clustering.onion_uni import onion_uni

#############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "onion_example_files/data/univariate_time-series.npy"

### Clustering at one specific value of time resolution ###
N_WINDOWS = 99

input_data = np.load(PATH_TO_INPUT_DATA)[:, 1:]
n_particles = input_data.shape[0]
n_frames = input_data.shape[1]

tau_window = int(n_frames / N_WINDOWS)
frames_in_excess = n_frames - N_WINDOWS * tau_window
reshaped_data = np.reshape(
    input_data[:, :-frames_in_excess], (n_particles * N_WINDOWS, tau_window)
)

state_list, labels = onion_uni(
    reshaped_data,
    n_windows=N_WINDOWS,
)

### These functions are examples of how to visualize the results
# plot_output_uni("Fig1.png", reshaped_data, N_WINDOWS, state_list)
# plot_one_trj_uni("Fig2.png", 1234, reshaped_data, labels, N_WINDOWS)
# plot_medoids_uni("Fig3.png", reshaped_data, labels)
# plot_state_populations("Fig4.png", N_WINDOWS, labels)
# plot_sankey("Fig5.png", labels, N_WINDOWS, [10, 20, 30, 40])

### Clustering at all the possible time resolution ###
TMP_LIST = np.geomspace(2, 249, num=20, dtype=int)
N_WINDOWS_LIST = [x for i, x in enumerate(TMP_LIST) if x not in TMP_LIST[:i]]

tra = np.zeros((len(N_WINDOWS_LIST), 3))
list_of_pop = []

for i, n_windows in enumerate(N_WINDOWS_LIST):
    tau_window = int(n_frames / n_windows)
    frames_in_excess = n_frames - n_windows * tau_window
    reshaped_data = np.reshape(
        input_data[:, :-frames_in_excess],
        (n_particles * n_windows, tau_window),
    )

    state_list, labels = onion_uni(
        reshaped_data,
        n_windows=n_windows,
    )

    pop_list = [state.perc for state in state_list]
    pop_list.insert(0, 1 - np.sum(np.array(pop_list)))
    list_of_pop.append(pop_list)

    tra[i][0] = tau_window
    tra[i][1] = len(state_list)
    tra[i][2] = pop_list[0]

### These functions are examples of how to visualize the results
plot_time_res_analysis("Fig6.png", tra)
plot_pop_fractions("Fig7.png", list_of_pop[::-1])

plt.show()
