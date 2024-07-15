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

    Attributes
    ----------

    iterations : int
        The number of iterations the algorithm performed.
    """

    def __init__(self, par: Parameters, data: Union[UniData, MultiData]):
        self.par = par
        self.data = data
        self.iterations = -1

    def create_all_the_labels(self) -> np.ndarray:
        """
        Assigns labels to the signal windows.

        Returns
        -------

        all_the_labels : np.ndarray
            An updated ndarray with labels assigned to individual frames
            by repeating the existing labels.

        """
        all_the_labels = np.reshape(self.data.labels, (self.data.labels.size,))
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
