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
    N_WINDOWS = 1000
    BINS = 50

    input_data = np.load(PATH_TO_INPUT_DATA)

    reshaped_data = np.reshape(input_data, (2 * N_WINDOWS, 2 * 10))

    # Call your code to generate the output files
    tmp = OnionMulti(
        n_windows=N_WINDOWS,
        bins=BINS,
    )
    tmp.fit_predict(reshaped_data)

    _, labels = onion_multi(
        reshaped_data,
        n_windows=N_WINDOWS,
        bins=BINS,
    )

    # Define the paths to the expected output files
    original_dir = (
        "/Users/mattebecchi/00_signal_analysis/timeseries_analysis/test/"
    )
    expected_output_path = original_dir + "output_multi/labels.npy"

    # np.save(expected_output_path, labels)

    # Compare the contents of the expected and actual output
    expected_output = np.load(expected_output_path)
    assert np.allclose(expected_output, labels)
