"""
Contains the classes used for storing parameters and system states.
"""

from typing import Union

import numpy as np


class StateUni:
    """
    Represents a unidimensional state as a Gaussian.

    Parameters
    ----------

    mean : float
        Mean of the Gaussian.

    sigma : float
        Rescaled standard deviation of the Gaussian.

    area : float
        Area below the Gaussian.

    Attributes
    ----------

    peak : float
        Maximum value of the Gaussian.

    perc : float
        Fraction of data points classified in the state.

    th_inf : float
        Lower thrashold of the state.

    th_sup : float
        Upper thrashold of the state.
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

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self.th_inf = [self.mean - number_of_sigmas * self.sigma, -1]
        self.th_sup = [self.mean + number_of_sigmas * self.sigma, -1]


class StateMulti:
    """
    Represents a multifimensional state as a factorized Gaussian.

    Parameters
    ----------

    mean : np.ndarray of shape (dim,)
        Mean of the Gaussians.

    sigma : np.ndarray of shape (dim,)
        Rescaled standard deviation of the Gaussians.

    area : np.ndarray of shape (dim,)
        Area below the Gaussians.

    Attributes
    ----------

    perc : float
        Fraction of data points classified in this state.

    axis : ndarray of shape (dim,)
        The thrasholds of the state.
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

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self.axis = number_of_sigmas * self.sigma  # Axes of the state


class UniData:
    """
    The input univariate signals to cluster.

    Parameters
    ----------

    matrix : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    Attributes
    ----------

    number_of_particles : int
        The number of particles in the system.

    num_of_steps : int
        The number of frames in the system.

    range : ndarray of shape (2,)
        Min and max of the signals.

    labels : ndarray of shape (n_particles, n_frames)
        The cluster labels.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.num_of_particles = self.matrix.shape[0]
        self.num_of_steps = self.matrix.shape[1]
        self.labels = np.array([])
        if matrix.size > 0:
            self.range = np.array([np.min(self.matrix), np.max(self.matrix)])


class MultiData:
    """
    The input mutivariate signals to cluster.

    Parameters
    ----------

    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    Attributes
    ----------

    dims : int
        The dimension of the space of the signals.

    number_of_particles : int
        The number of particles in the system.

    num_of_steps : int
        The number of frames in the system.

    range : ndarray of shape (dim, 2)
        Min and max of the signals along each axes.

    matrix : ndarray of shape (n_particles, n_frames, dims)
        The values of the signal for each particle at each frame.

    labels : ndarray of shape (n_particles, n_frames)
        The cluster labels.
    """

    def __init__(self, matrix: np.ndarray):
        self.dims = matrix.shape[0]
        self.num_of_particles = matrix.shape[1]
        self.num_of_steps = matrix.shape[2]
        self.range = np.array(
            [[np.min(data), np.max(data)] for data in matrix]
        )

        self.matrix: np.ndarray = np.transpose(matrix, axes=(1, 2, 0))
        self.labels = np.array([])


class Parameters:
    """
    Contains the set of parameters for the specific analysis.

    Parameters
    ----------

    tau_w : int
        The time resolution for the clustering, corresponding to the length
        of the windows in which the time-series are segmented.

    bins: Union[str, int]
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/
        numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).
    """

    def __init__(
        self,
        tau_window: int,
        bins: Union[int, str],
        number_of_sigmas: float,
    ):
        self.tau_w = tau_window
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas
