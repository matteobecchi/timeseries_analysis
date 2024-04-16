"""
Contains the classes used for storing the clustering data.
"""

from typing import List, Union

import numpy as np
from onion_clustering._internal.first_classes import (
    MultiData,
    Parameters,
    StateMulti,
    StateUni,
    UniData,
)


class ClusteringObject:
    """
    This class contains the cluster's input and output.

    Parameters
    ----------

    par : Parameters
        The parameters of the analysis.

    data : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window_list : List[int]
        The list of time resolutions at which the fast analysis will
        be performed. If None (default), use a logspaced list between 2 and
        the entire trajectory length.

    Attributes
    ----------

    iterations : int
        The number of iterations the algorithm performed.

    number_of_states : np.ndarray
        For every value of tau_window_list.

    fraction_0 : np.ndarray
        For every value of tau_window_list.
    """

    def __init__(self, par: Parameters, data: Union[UniData, MultiData]):
        self.par = par
        self.data = data
        self.tau_window_list: List[int]
        self.iterations = -1
        self.number_of_states: np.ndarray
        self.fraction_0: np.ndarray

    def create_all_the_labels(self) -> np.ndarray:
        """
        Assigns labels to individual frames by repeating the existing labels.

        Returns
        -------

        all_the_labels : np.ndarray
            An updated ndarray with labels assigned to individual frames
            by repeating the existing labels.

        """
        all_the_labels = np.repeat(self.data.labels, self.par.tau_w, axis=1)
        return all_the_labels


class ClusteringObject1D(ClusteringObject):
    """
    This class contains the cluster's input and output.

    Attributes
    ----------

    state_list : List[StateUni]
        The list of states found during the clustering.
    """

    state_list: List[StateUni] = []


class ClusteringObject2D(ClusteringObject):
    """
    This class contains the cluster's input and output.

    Attributes
    ----------

    state_list : List[StateMulti]
        The list of states found during the clustering.
    """

    state_list: List[StateMulti] = []
