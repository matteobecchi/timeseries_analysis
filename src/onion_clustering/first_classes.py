"""
Contains the classes used for storing parameters and system states.
"""

import copy

import numpy as np


class StateUni:
    """
    Represents a unidimensional state as a Gaussian.

    Attributes:
    - mean (float): mean of the Gaussian
    - sigma (float): rescaled standard deviation of the Gaussian
    - area (float): area below the Gaussian
    - peak (float): height of the Gaussian peak
    - perc (float): fraction of data points classified in this state
    - th_inf (float): lower thrashold of the state
    - th_sup (float): upper thrashold of the state
    """

    def __init__(self, mean: float, sigma: float, area: float):
        self.mean = mean
        self.sigma = sigma
        self.area = area
        self.peak = area / sigma / np.sqrt(np.pi)
        self.perc = 0.0
        self.th_inf = [mean - 2.0 * sigma, -1]
        self.th_sup = [mean + 2.0 * sigma, -1]

    def build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Args:
        - number of sigmas (float)
        """
        self.th_inf = [self.mean - number_of_sigmas * self.sigma, -1]
        self.th_sup = [self.mean + number_of_sigmas * self.sigma, -1]


class StateMulti:
    """
    Represents a multifimensional state as a factorized Gaussian.

    Attributes:
    - mean (np.ndarray): mean of the Gaussians
    - sigma (np.ndarray): rescaled standard deviation of the Gaussians
    - area (np.ndarray): area below the Gaussians
    - perc (float): fraction of data points classified in this state
    - axis: the thrasholds of the state
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, area: np.ndarray):
        self.mean = mean
        self.sigma = sigma
        self.area = area
        self.perc = 0.0
        self.axis = 2.0 * sigma

    def build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Args:
        - number of sigmas (float)
        """
        self.axis = number_of_sigmas * self.sigma  # Axes of the state


class UniData:
    """
    The input univariate signals to cluster.

    Attributes:
    - matrix (np.ndarray): the (N, T) array with the signals
    - number_of_particles (int)
    - num_of_steps (int)
    - range (np.ndarray): min and max of the signals
    - labels (np.ndarray): the clustering output
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.num_of_particles = self.matrix.shape[0]
        self.num_of_steps = self.matrix.shape[1]
        self.range = np.array([np.min(self.matrix), np.max(self.matrix)])
        self.labels = np.array([])

    def print_info(self):
        """
        Prints information about the input data.
        """
        print("Number of particles:", self.num_of_particles)
        print("Number of steps:", self.num_of_steps)
        print("Data range:", self.range)

    def create_copy(self):
        """
        Returns an independent copy of the UniData object.

        Changes to the copy will not affect the original object
        """
        copy_data = copy.deepcopy(self)
        return copy_data


class MultiData:
    """
    The input signals of the analysis.
    """

    def __init__(self, matrix: np.ndarray):
        for i, component in enumerate(matrix[:-1]):
            if component.shape != matrix[i + 1].shape:
                print("ERROR: The signals do not correspond. Abort.")
                return
        self.dims = matrix.shape[0]
        self.num_of_particles = matrix.shape[1]
        self.num_of_steps = matrix.shape[2]
        self.range = np.array(
            [[np.min(data), np.max(data)] for data in matrix]
        )

        self.matrix: np.ndarray = np.transpose(matrix, axes=(1, 2, 0))
        self.labels = np.array([])

    def print_info(self):
        """
        Prints information about the input data.
        """
        print("Number of particles:", self.num_of_particles)
        print("Number of steps:", self.num_of_steps)
        print("Number of components:", self.dims)
        print("Data range:", self.range)

    def remove_delay(self, t_delay: int):
        """
        Remove the first t_delay points from the trjs.

        Args:
        - t_delay (int): number of steps to remove from the beginning
        of the trjs
        """
        self.matrix = self.matrix[:, t_delay:, :]
        self.num_of_steps = self.matrix.shape[1]

    def smooth(self, window: int):
        """
        Smooths the data using a moving average with a specified window size.

        Args:
        - window (int): Size of the moving average window
        """
        weights = np.ones(window) / window
        tmp_matrix = np.transpose(self.matrix, axes=(2, 0, 1))
        tmp_matrix = np.apply_along_axis(
            lambda x: np.convolve(x, weights, mode="valid"),
            axis=2,
            arr=tmp_matrix,
        )
        self.matrix = np.transpose(tmp_matrix, axes=(1, 2, 0))
        self.num_of_steps = self.matrix.shape[1]
        self.range = np.array(
            [[np.min(comp), np.max(comp)] for comp in tmp_matrix]
        )

    def normalize(self, dim_to_avoid: list[int]):
        """Normalizes linearly the data between 0 and 1."""
        tmp_matrix = np.transpose(self.matrix, axes=(2, 0, 1))
        new_matrix = []
        for dim, comp in enumerate(tmp_matrix):
            if dim not in dim_to_avoid:
                data_min, data_max = np.min(comp), np.max(comp)
                new_matrix.append((comp - data_min) / (data_max - data_min))
            else:
                new_matrix.append(comp)
        self.matrix = np.transpose(np.array(new_matrix), axes=(1, 2, 0))
        self.range = np.array(
            [[np.min(comp), np.max(comp)] for comp in new_matrix]
        )

    def create_copy(self):
        """
        Returns an independent copy of the UniData object.
        Changes to the copy will not affect the original object.
        """
        copy_data = copy.deepcopy(self)
        return copy_data


class Parameters:
    """
    Contains the set of parameters for the specific analysis.

    Attributes:
    - tau_w (int): time resolution of the analysis
    - t_smooth (int): smoothing window
    - t_delay (int): time steps to remove from the beginning
    - t_conv (float): conversion factro between frames and time units
    - t_units (str): time units
    - example_id (int): selected example particle
    - bins (str/int): method for binning / number of bins for the histogram
    - num_tau_w (int): number of time resolutoin to use
    - min_tau_w (int): minimum time resolution to use
    - max_tau_w (int): maximum time resolution to use
    - step_t_smooth (int): number of smoothing window to use
    - min_t_smooth (int): minimum smoothing window to use
    - max_t_smooth (int): maximum smoothing window to use
    """

    def __init__(self, tau_window, bins, num_tau_w, min_tau_w, max_tau_w):
        self.tau_w = tau_window
        self.bins = bins
        self.num_tau_w = num_tau_w
        self.min_tau_w = min_tau_w
        self.max_tau_w = max_tau_w

    def create_copy(self):
        """
        Returns an independent copy of the Parameter object.
        Changes to the copy will not affect the original object.
        """
        copy_par = copy.deepcopy(self)
        return copy_par
