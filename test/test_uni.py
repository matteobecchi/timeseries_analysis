import os

import numpy as np
import pytest
from onion_clustering.onion_uni import OnionUni, onion_uni


# Define a fixture to set up the test environment
@pytest.fixture
def setup_test_environment(tmpdir):
    # tmpdir is a built-in pytest fixture providing a temporary directory
    original_dir = os.getcwd()  # Save the current working directory
    os.chdir(str(tmpdir))  # Change to the temporary directory
    yield tmpdir
    os.chdir(
        original_dir
    )  # Restore the original working directory after the test


# Define the actual test
def test_output_files(setup_test_environment):
    ### Set all the analysis parameters ###
    FILE = "water_coex_100ps_1nm_LENS.npy"
    PATH_TO_INPUT_DATA = "/Users/mattebecchi/00_signal_analysis/data/" + FILE
    TAU_WINDOW = 10  # time resolution of the analysis
    NUM_TAU_W = 2
    MAX_TAU_W = 10

    input_data = np.load(PATH_TO_INPUT_DATA)

    # Call your code to generate the output files
    tmp = OnionUni(
        tau_window=TAU_WINDOW, max_tau_w=MAX_TAU_W, num_tau_w=NUM_TAU_W
    )
    tmp.fit_predict(input_data)

    _, labels, time_res_analysis = onion_uni(
        input_data,
        tau_window=TAU_WINDOW,
        num_tau_w=NUM_TAU_W,
        max_tau_w=MAX_TAU_W,
    )

    # Define the paths to the expected output files
    original_dir = (
        "/Users/mattebecchi/00_signal_analysis/timeseries_analysis/test/"
    )
    expected_output_path_1 = original_dir + "output_uni/labels.npy"
    expected_output_path_2 = original_dir + "output_uni/time_res_analysis.txt"

    # Compare the contents of the expected and actual output
    expected_output_1 = np.load(expected_output_path_1)
    assert np.allclose(expected_output_1, labels)

    expected_output_2 = np.loadtxt(expected_output_path_2)
    assert np.allclose(expected_output_2, time_res_analysis)
