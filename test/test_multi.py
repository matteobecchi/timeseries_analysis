import os

import numpy as np
import pytest
from onion_clustering.onion_multi import OnionMulti, onion_multi


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
    FILE = "data/synthetic_2D/3D_synthetic_data.npy"
    PATH_TO_INPUT_DATA = "/Users/mattebecchi/00_signal_analysis/" + FILE
    TAU_WINDOW = 10  # time resolution of the analysis
    NUM_TAU_W = 2  # number of tau_window tested (default 20)
    MAX_TAU_W = 10  # max value of tau_window tested (default auto)
    BINS = 50  # number of histogram bins (default auto)

    input_data = np.load(PATH_TO_INPUT_DATA)

    # Call your code to generate the output files
    tmp = OnionMulti(
        tau_window=TAU_WINDOW,
        max_tau_w=MAX_TAU_W,
        num_tau_w=NUM_TAU_W,
        bins=BINS,
    )
    tmp.fit_predict(input_data)

    _, labels, time_res_analysis = onion_multi(
        input_data,
        tau_window=TAU_WINDOW,
        num_tau_w=NUM_TAU_W,
        max_tau_w=MAX_TAU_W,
        bins=BINS,
    )

    # Define the paths to the expected output files
    original_dir = (
        "/Users/mattebecchi/00_signal_analysis/timeseries_analysis/test/"
    )
    expected_output_path_1 = original_dir + "output_multi/labels.npy"
    expected_output_path_2 = (
        original_dir + "output_multi/time_res_analysis.txt"
    )

    # Compare the contents of the expected and actual output
    expected_output_1 = np.load(expected_output_path_1)
    assert np.allclose(expected_output_1, labels)

    expected_output_2 = np.loadtxt(expected_output_path_2)
    assert np.allclose(expected_output_2, time_res_analysis)
