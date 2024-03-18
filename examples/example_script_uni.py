"""
Example script for running onion_clustering
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from onion_clustering.onion_uni import onion_uni

#############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "onion_example_files/data/univariate_time-series.npy"
TAU_WINDOW = 10  # time resolution of the analysis

### Optional parametrers ###
NUM_TAU_W = 10  # number of values of tau_window tested (default 20)
MIN_TAU_W = 2  # min number of tau_window tested (default 2)
#############################################################################

input_data = np.load(PATH_TO_INPUT_DATA)

state_list, labels, time_res_analysis = onion_uni(
    input_data, tau_window=TAU_WINDOW, num_tau_w=NUM_TAU_W, min_tau_w=MIN_TAU_W
)

### Plot the number of states and fraction of unclassified data points
### as a function of the time resolution
fig, axes = plt.subplots()
axes.plot(time_res_analysis[:, 0], time_res_analysis[:, 1], marker="o")
axes.set_xlabel(r"Time resolution $\Delta t$ [frame]")
axes.set_ylabel(r"# environments", weight="bold", c="#1f77b4")
axes.set_xscale("log")
axes.yaxis.set_major_locator(MaxNLocator(integer=True))
axesr = axes.twinx()
axesr.plot(
    time_res_analysis[:, 0], time_res_analysis[:, 2], marker="o", c="#ff7f0e"
)
axesr.set_ylabel("Population of env 0", weight="bold", c="#ff7f0e")
plt.show()
