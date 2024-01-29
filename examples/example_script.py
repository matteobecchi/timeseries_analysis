"""
Example script for running onion_clustering
"""
from onion_clustering import main

###############################################################################################
### Set all the analysis parameters ###
PATH_TO_INPUT_DATA = 'your/data/directory/here/dataset.npy'
TAU_WINDOW = 10         # time resolution of the analysis

### Optional parametrers ###
T_SMOOTH = 1            # window for moving average (default 1)
T_DELAY = 0             # remove the first t_delay frames (default 0)
T_CONV = 1.             # convert frames in time units (default 1)
TIME_UNITS = 'frames'   # the time units (default 'frames')
EXAMPLE_ID = 0          # particle plotted as example (default 0)
NUM_TAU_W = 20          # number of values of tau_window tested (default 20)
MIN_TAU_W = 2           # min number of tau_window tested (default 2)
MIN_T_SMOOTH = 1        # min value of t_smooth tested (default 1)
MAX_T_SMOOTH = 5        # max value of t_smooth tested (default 5)
STEP_T_SMOOTH = 1       # increment in value of t_smooth tested (default 1)
MAX_TAU_W = 'auto'      # max number of tau_window tested (default is automatic)
BINS = 'auto'           # number of histogram bins (default is automatic)
###############################################################################################

### Create the 'data_directory.txt' file ###
with open('data_directory.txt', "w+", encoding="utf-8") as file:
    print(PATH_TO_INPUT_DATA, file=file)

### Create the 'input_parameter.txt' file ###
with open('input_parameters.txt', "w+", encoding="utf-8") as file:
    print('tau_window\t' + str(TAU_WINDOW), file=file)
    print('t_smooth\t' + str(T_SMOOTH), file=file)
    print('t_delay\t' + str(T_DELAY), file=file)
    print('t_conv\t' + str(T_CONV), file=file)
    print('t_units\t' + TIME_UNITS, file=file)
    print('example_ID\t' + str(EXAMPLE_ID), file=file)
    print('num_tau_w\t' + str(NUM_TAU_W), file=file)
    print('min_tau_w\t' + str(MIN_TAU_W), file=file)
    print('min_t_smooth\t' + str(MIN_T_SMOOTH), file=file)
    print('max_t_smooth\t' + str(MAX_T_SMOOTH), file=file)
    print('step_t_smooth\t' + str(STEP_T_SMOOTH), file=file)
    if MAX_TAU_W != 'auto':
        print('max_tau_w\t' + str(MAX_TAU_W), file=file)
    if BINS != 'auto':
        print('bins\t' + str(BINS), file=file)

### Run the code ###
main.main()