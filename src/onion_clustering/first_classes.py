"""
Contains the classes used for storing parameters and system states.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex
from matplotlib.ticker import MaxNLocator

COLORMAP = "viridis"


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

    def plot_medoids(self):
        """
        Compute and plot the average signal sequence inside each state.

        - Initializes some usieful variables
        - Check if we need to add the "0" state for consistency
        - For each state, stores all the signal windows in that state, and
        - Computes mean and standard deviation of the signals in that state
        - Prints the output to file
        - Plots the results to Fig4.png

        """
        tau_window = int(self.num_of_steps / self.labels.shape[1])
        all_the_labels = self.labels
        center_list = []
        std_list = []

        missing_zero = 0
        list_of_labels = np.unique(all_the_labels)
        if 0 not in list_of_labels:
            list_of_labels = np.insert(list_of_labels, 0, 0)
            missing_zero = 1

        for ref_label in list_of_labels:
            tmp = []
            for i, mol in enumerate(all_the_labels):
                for window, label in enumerate(mol):
                    time_0 = window * tau_window
                    time_1 = (window + 1) * tau_window
                    if label == ref_label:
                        tmp.append(self.matrix[i][time_0:time_1])
            if len(tmp) > 0:
                center_list.append(np.mean(tmp, axis=0))
                std_list.append(np.std(tmp, axis=0))
        center_arr = np.array(center_list)
        std_arr = np.array(std_list)

        np.savetxt(
            "medoid_center.txt",
            center_arr,
            header="Signal average for each ENV",
        )
        np.savetxt(
            "medoid_stddev.txt",
            std_arr,
            header="Signal standard deviation for each ENV",
        )

        palette = []
        cmap = plt.get_cmap(COLORMAP, list_of_labels.size)
        palette.append(rgb2hex(cmap(0)))
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))
        fig, axes = plt.subplots()
        time_seq = range(tau_window)
        for center_id, center in enumerate(center_list):
            err_inf = center - std_list[center_id]
            err_sup = center + std_list[center_id]
            axes.fill_between(
                time_seq,
                err_inf,
                err_sup,
                alpha=0.25,
                color=palette[center_id + missing_zero],
            )
            axes.plot(
                time_seq,
                center,
                label="ENV" + str(center_id + missing_zero),
                marker="o",
                c=palette[center_id + missing_zero],
            )
        fig.suptitle("Average time sequence inside each environments")
        axes.set_xlabel(r"Time $t$ [frames]")
        axes.set_ylabel(r"Signal")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.legend()
        fig.savefig("output_figures/Fig4.png", dpi=600)


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

    def plot_medoids(self):
        """
        Compute and plot the average signal sequence inside each state.

        - Checks if the data have 2 or 3 components (works only with D == 2)
        - Initializes some usieful variables
        - Check if we need to add the "0" state for consistency
        - For each state, stores all the signal windows in that state, and
        - Computes mean of the signals in that state
        - Prints the output to file
        - Plots the results to Fig4.png

        """
        if self.dims > 2:
            print("plot_medoids() does not work with 3D data.")
            return

        missing_zero = 0
        list_of_labels = np.unique(self.labels)
        if 0 not in list_of_labels:
            list_of_labels = np.insert(list_of_labels, 0, 0)
            missing_zero = 1

        tau_window = int(self.num_of_steps / self.labels.shape[1])
        center_list = []

        for ref_label in list_of_labels:
            tmp = []
            for i, mol in enumerate(self.labels):
                for j, label in enumerate(mol):
                    t_0 = j * tau_window
                    t_1 = (j + 1) * tau_window
                    if label == ref_label:
                        tmp.append(self.matrix[i][t_0:t_1])
            center_list.append(np.mean(tmp, axis=0))

        palette = []
        cmap = plt.get_cmap(COLORMAP, np.unique(self.labels).size)
        palette.append(rgb2hex(cmap(0)))
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))
        fig, axes = plt.subplots()
        for id_c, center in enumerate(center_list):
            sig_x = center[:, 0]
            sig_y = center[:, 1]
            axes.plot(
                sig_x,
                sig_y,
                label="ENV" + str(id_c + missing_zero),
                marker="o",
                c=palette[id_c + missing_zero],
            )
        fig.suptitle("Average time sequence inside each environments")
        axes.set_xlabel(r"Signal 1")
        axes.set_ylabel(r"Signal 2")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.legend()
        fig.savefig("output_figures/Fig4.png", dpi=600)


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
