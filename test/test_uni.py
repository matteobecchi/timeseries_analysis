import os
import filecmp
import pytest
from onion_clustering import main

# Define a fixture to set up the test environment
@pytest.fixture
def setup_test_environment(tmpdir):
    # tmpdir is a built-in pytest fixture providing a temporary directory
    original_dir = os.getcwd()  # Save the current working directory
    os.chdir(str(tmpdir))  # Change to the temporary directory
    yield tmpdir
    os.chdir(original_dir)  # Restore the original working directory after the test

# Define the actual test
def test_output_files(setup_test_environment):
    ### Set all the analysis parameters ###
    PATH_TO_INPUT_DATA = '/Users/mattebecchi/00_signal_analysis/data/water_coex_100ps_1nm_LENS.npy'
    TAU_WINDOW = 10         # time resolution of the analysis
    T_DELAY = 1             # remove the first t_delay frames (default 0)
    T_CONV = 0.1            # convert frames in time units (default 1)
    T_UNITS = 'ns'          # the time units (default 'frames')
    NUM_TAU_W = 2          
    MAX_TAU_W = 10          
    MAX_T_SMOOTH = 2        # max value of t_smooth tested (default 5)

    ### Create the 'data_directory.txt' file ###
    with open('data_directory.txt', "w+", encoding="utf-8") as file:
        print(PATH_TO_INPUT_DATA, file=file)

    ### Create the 'input_parameter.txt' file ###
    with open('input_parameters.txt', "w+", encoding="utf-8") as file:
        print('tau_window\t' + str(TAU_WINDOW), file=file)
        print('t_delay\t' + str(T_DELAY), file=file)
        print('t_conv\t' + str(T_CONV), file=file)
        print('t_units\t' + T_UNITS, file=file)
        print('num_tau_w\t' + str(NUM_TAU_W), file=file)
        print('max_tau_w\t' + str(MAX_TAU_W), file=file)
        print('max_t_smooth\t' + str(MAX_T_SMOOTH), file=file)

    # Call your code to generate the output files
    main.main()

    # Define the paths to the expected and actual output files
    original_dir = "/Users/mattebecchi/00_signal_analysis/timeseries_analysis/test/"
    expected_output_path_1 = original_dir + "output_uni/final_states.txt"
    expected_output_path_2 = original_dir + "output_uni/number_of_states.txt"
    expected_output_path_3 = original_dir + "output_uni/fraction_0.txt"
    actual_output_path_1 = "final_states.txt"
    actual_output_path_2 = "number_of_states.txt"
    actual_output_path_3 = "fraction_0.txt"

    # Use filecmp to compare the contents of the expected and actual output directories
    with open(expected_output_path_1, 'r') as expected_file, open(actual_output_path_1, 'r') as actual_file:
        assert expected_file.read() == actual_file.read()
    with open(expected_output_path_2, 'r') as expected_file, open(actual_output_path_2, 'r') as actual_file:
        assert expected_file.read() == actual_file.read()
    with open(expected_output_path_3, 'r') as expected_file, open(actual_output_path_3, 'r') as actual_file:
        assert expected_file.read() == actual_file.read()