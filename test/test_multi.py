import os
import filecmp
import pytest
from onion_clustering import main_2d

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
    PATH_TO_INPUT_DATA_0 = '/Users/mattebecchi/00_signal_analysis/synthetic_2D/3D_synthetic_data_0.npy'
    PATH_TO_INPUT_DATA_1 = '/Users/mattebecchi/00_signal_analysis/synthetic_2D/3D_synthetic_data_1.npy'
    TAU_WINDOW = 10         # time resolution of the analysis
    T_CONV = 200            # convert frames in time units (default 1)
    T_UNITS = 'dt'          # the time units (default 'frames')
    NUM_TAU_W = 2           # number of tau_window tested (default 20)
    MAX_TAU_W = 10          # max value of tau_window tested (default auto)
    MAX_T_SMOOTH = 2        # max value of t_smooth tested (default 5)
    BINS = 50               # number of histogram bins (default auto)

    ### Create the 'data_directory.txt' file ###
    with open('data_directory.txt', "w+", encoding="utf-8") as file:
        print(PATH_TO_INPUT_DATA_0, file=file)
        print(PATH_TO_INPUT_DATA_1, file=file)

    ### Create the 'input_parameter.txt' file ###
    with open('input_parameters.txt', "w+", encoding="utf-8") as file:
        print('tau_window\t' + str(TAU_WINDOW), file=file)
        print('t_conv\t' + str(T_CONV), file=file)
        print('t_units\t' + T_UNITS, file=file)
        print('num_tau_w\t' + str(NUM_TAU_W), file=file)
        print('max_tau_w\t' + str(MAX_TAU_W), file=file)
        print('max_t_smooth\t' + str(MAX_T_SMOOTH), file=file)
        print('bins\t' + str(BINS), file=file)

    # Call your code to generate the output files
    main_2d.main()

    # Define the paths to the expected and actual output files
    original_dir = "/Users/mattebecchi/00_signal_analysis/timeseries_analysis/test/"
    expected_output_path_1 = original_dir + "output_multi/final_states.txt"
    expected_output_path_2 = original_dir + "output_multi/number_of_states.txt"
    expected_output_path_3 = original_dir + "output_multi/fraction_0.txt"
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
