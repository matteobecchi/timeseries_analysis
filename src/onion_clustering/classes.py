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


class ClusteringObject1D(ClusteringObject):
    """This class contains the cluster's input and output."""

    state_list: List[StateUni] = []


class ClusteringObject2D(ClusteringObject):
    """This class contains the cluster's input and output."""

    state_list: List[StateMulti] = []
