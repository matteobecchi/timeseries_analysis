"""
Contains the classes used for storing the clustering data.
"""

from typing import List, Union

import numpy as np

from onion_clustering.first_classes import (
    MultiData,
    Parameters,
    StateMulti,
    StateUni,
    UniData,
)


class ClusteringObject:
    """
    This class contains the cluster's input and output.

    Attributes
    ----------

    par (Parameters): the parameters of the analysis

    data (np.ndarray): the data points

    iterations (int): the number of iterations the algorithm performed

    number_of_states (np.ndarray): for every pair of tau_w and t_smooth

    fraction_0 (np.ndarray): for every pair of tau_w and t_smooth
    """

    def __init__(self, par: Parameters, data: Union[UniData, MultiData]):
        self.par = par
        self.data = data
        self.iterations = -1
        self.number_of_states: np.ndarray
        self.fraction_0: np.ndarray
        self.tau_window_list: List[int]

    def create_all_the_labels(self) -> np.ndarray:
        """
        Assigns labels to individual frames by repeating the existing labels.

        Returns:
        - np.ndarray: an updated ndarray with labels assigned
        to individual frames by repeating the existing labels

        """
        all_the_labels = np.repeat(self.data.labels, self.par.tau_w, axis=1)
        return all_the_labels

    def print_labels(self):
        """
        Print the label for every particle for every frame.
        Output is a (N, T) array in a .npy file.
        """
        print("* Print labels for all the data points...")
        all_the_labels = self.create_all_the_labels()
        np.save("all_labels.npy", all_the_labels)

    def print_signal_with_labels(self):
        """
        Creates a file ('signal_with_labels.dat') with signal values
        and associated cluster labels.
        """
        m_clean = self.data.matrix
        all_the_labels = self.create_all_the_labels()

        with open("signal_with_labels.dat", "w+", encoding="utf-8") as file:
            if m_clean.shape[2] == 2:
                print("Signal 1 Signal 2 Cluster Frame", file=file)
            else:
                print("Signal 1 Signal 2 Signal 3 Cluster Frame", file=file)
            for j in range(all_the_labels.shape[1]):
                for i in range(all_the_labels.shape[0]):
                    if m_clean.shape[2] == 2:
                        print(
                            m_clean[i][j][0],
                            m_clean[i][j][1],
                            all_the_labels[i][j],
                            j + 1,
                            file=file,
                        )
                    else:
                        print(
                            m_clean[i][j][0],
                            m_clean[i][j][1],
                            m_clean[i][j][2],
                            all_the_labels[i][j],
                            j + 1,
                            file=file,
                        )


class ClusteringObject1D(ClusteringObject):
    """This class contains the cluster's input and output."""

    state_list: List[StateUni] = []


class ClusteringObject2D(ClusteringObject):
    """This class contains the cluster's input and output."""

    state_list: List[StateMulti] = []

    def preparing_the_data(self):
        """
        Prepare the raw data for analysis.

        This function prepares the raw data for analysis:
        - Applies a moving average filter on the raw data.
        - Normalizes the data to the range [0, 1] (commented out in the code).
        - Calculates the number of windows for analysis based on parameters.
        - Prints informative messages about trajectory details.

        """
        tau_window, t_smooth = self.par.tau_w, self.par.t_smooth
        t_conv, t_units = self.par.t_conv, self.par.t_units

        self.data.smooth(t_smooth)
        ### Normalizes data in [0, 1]. Usually not necessary.
        ### The arg is the list of components to NOT normalize
        # self.data.normalize([])

        # Calculate the number of windows for the analysis.
        num_windows = int(self.data.num_of_steps / tau_window)

        # Print informative messages about trajectory details.
        print(
            "\tTrajectory has "
            + str(self.data.num_of_particles)
            + " particles. "
        )
        print(
            "\tTrajectory of length "
            + str(self.data.num_of_steps)
            + " frames ("
            + str(self.data.num_of_steps * t_conv)
            + " "
            + t_units
            + ")."
        )
        print(
            "\tUsing "
            + str(num_windows)
            + " windows of length "
            + str(tau_window)
            + " frames ("
            + str(tau_window * t_conv)
            + " "
            + t_units
            + ")."
        )
