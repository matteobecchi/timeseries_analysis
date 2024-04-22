"""
Example script for running onion_clustering
"""

import os
import shutil
from pathlib import Path

from onion_clustering import main

##############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "../onion_example_files/data/univariate_time-series.npy"
TAU_WINDOW = 10  # time resolution of the analysis

### Optional parametrers ###
T_SMOOTH = 1  # window for moving average (default 1)
T_DELAY = 1  # remove the first t_delay frames (default 0)
T_CONV = 1.0  # convert frames in time units (default 1)
TIME_UNITS = "frames"  # the time units (default 'frames')
EXAMPLE_ID = 0  # particle plotted as example (default 0)
NUM_TAU_W = 10  # number of values of tau_window tested (default 20)
MIN_TAU_W = 2  # min number of tau_window tested (default 2)
MIN_T_SMOOTH = 1  # min value of t_smooth tested (default 1)
MAX_T_SMOOTH = 2  # max value of t_smooth tested (default 5)
STEP_T_SMOOTH = 1  # increment in value of t_smooth tested (default 1)
MAX_TAU_W = "auto"  # max number of tau_window tested (default is auto)
BINS = "auto"  # number of histogram bins (default is auto)
##############################################################################

### Create the output directory and move there ###
original_dir = Path.cwd()
output_path = Path("./onion_output")
if output_path.exists():
    shutil.rmtree(output_path)
output_path.mkdir()
os.chdir(output_path)

try:
    ### Create the 'data_directory.txt' file ###
    with open("data_directory.txt", "w+", encoding="utf-8") as file:
        print(PATH_TO_INPUT_DATA, file=file)

    ### Create the 'input_parameter.txt' file ###
    with open("input_parameters.txt", "w+", encoding="utf-8") as file:
        print("tau_window\t" + str(TAU_WINDOW), file=file)
        print("t_smooth\t" + str(T_SMOOTH), file=file)
        print("t_delay\t" + str(T_DELAY), file=file)
        print("t_conv\t" + str(T_CONV), file=file)
        print("t_units\t" + TIME_UNITS, file=file)
        print("example_ID\t" + str(EXAMPLE_ID), file=file)
        print("num_tau_w\t" + str(NUM_TAU_W), file=file)
        print("min_tau_w\t" + str(MIN_TAU_W), file=file)
        print("min_t_smooth\t" + str(MIN_T_SMOOTH), file=file)
        print("max_t_smooth\t" + str(MAX_T_SMOOTH), file=file)
        print("step_t_smooth\t" + str(STEP_T_SMOOTH), file=file)
        if MAX_TAU_W != "auto":
            print("max_tau_w\t" + str(MAX_TAU_W), file=file)
        if BINS != "auto":
            print("bins\t" + str(BINS), file=file)

    ### Perform the clustering analysis ###
    cl_ob = main.main()

    ### Plot the output figures in output_figures/ ###

    # Plots number of states and fraction_0 as a function of tau_window
    cl_ob.plot_tra_figure()

    # Plots the raw data
    cl_ob.plot_input_data("Fig0")

    # Plots the data with the clustering thresholds and Gaussians
    cl_ob.plot_cumulative_figure()

    # Plots the colored signal for the particle with `example_ID` ID
    cl_ob.plot_one_trajectory()

    # Plots the mean time sequence inside each state
    cl_ob.data.plot_medoids()

    # Plots the population of each state as a function of time
    cl_ob.plot_state_populations()

    # Plots the Sankey diagram between the input time_windows
    cl_ob.sankey([0, 10, 20, 30, 40])

    # Writes the files for the visualization of the colored trj
    if os.path.exists("../trajectory.xyz"):
        cl_ob.print_colored_trj_from_xyz("../trajectory.xyz")
    else:
        cl_ob.print_labels()
finally:
    os.chdir(original_dir)
