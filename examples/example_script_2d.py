"""
Example script for running onion_clustering
"""

import os
import shutil
from pathlib import Path

from onion_clustering import main_2d

##############################################################################
### Set all the analysis parameters ###
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
PATH_TO_INPUT_DATA = "../onion_example_files/data/multivariate_time-series.npy"
TAU_WINDOW = 10  # time resolution of the analysis

### Optional parametrers ###
T_SMOOTH = 1  # window for moving average (default 1)
T_DELAY = 0  # remove the first t_delay frames (default 0)
T_CONV = 1.0  # convert frames in time units (default 1)
TIME_UNITS = "frames"  # the time units (default 'frames')
EXAMPLE_ID = 0  # particle plotted as example (default 0)
NUM_TAU_W = 20  # number of values of tau_window tested (default 20)
MIN_TAU_W = 2  # min number of tau_window tested (default 2)
MIN_T_SMOOTH = 1  # min value of t_smooth tested (default 1)
MAX_T_SMOOTH = 2  # max value of t_smooth tested (default 5)
STEP_T_SMOOTH = 1  # increment in value of t_smooth tested (default 1)
MAX_TAU_W = "auto"  # max number of tau_window tested (default is automatic)
BINS = "auto"  # number of histogram bins (default is automatic)
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
        print(f"tau_window\t{TAU_WINDOW}", file=file)
        print(f"t_smooth\t{T_SMOOTH}", file=file)
        print(f"t_delay\t{T_DELAY}", file=file)
        print(f"t_conv\t{T_CONV}", file=file)
        print(f"t_units\t{TIME_UNITS}", file=file)
        print(f"example_ID\t{EXAMPLE_ID}", file=file)
        print(f"num_tau_w\t{NUM_TAU_W}", file=file)
        print(f"min_tau_w\t{MIN_TAU_W}", file=file)
        print(f"min_t_smooth\t{MIN_T_SMOOTH}", file=file)
        print(f"max_t_smooth\t{MAX_T_SMOOTH}", file=file)
        print(f"step_t_smooth\t{STEP_T_SMOOTH}", file=file)
        if MAX_TAU_W != "auto":
            print(f"max_tau_w\t{MAX_TAU_W}", file=file)
        if BINS != "auto":
            print(f"bins\t{BINS}", file=file)

    ### Run the code ###
    cl_ob = main_2d.main()

    ### Plot the output figures in output_figures/ ###
    # Plots number of states and fraction_0 as a function of tau_window
    cl_ob.plot_tra_figure()
    cl_ob.plot_pop_fractions()

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
    cl_ob.sankey([0, 100, 200, 300, 400])

    # Writes the files for the visualization of the colored trj
    if os.path.exists("../trajectory.xyz"):
        cl_ob.print_colored_trj_from_xyz("../trajectory.xyz")
    else:
        cl_ob.print_labels()
finally:
    os.chdir(original_dir)
