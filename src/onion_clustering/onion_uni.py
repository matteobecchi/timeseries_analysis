"""onion-clustering for univariate time-series."""

from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from onion_clustering._internal.main import main as onion_inner


def onion_uni(
    matrix,
    tau_window: int = 2,
    tau_window_list: Union[List[int], None] = None,
    bins: Union[str, int] = "auto",
    number_of_sigmas: float = 2.0,
):
    """Perform onion clustering from data array.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering, corresponding to the length
        of the windows in which the time-series are segmented.

    tau_window_list : List[int]
        The list of time resolutions at which the fast analysis will
        be performed. If None (default), use a logspaced list between 2 and
        the entire trajectory length.

    bins: Union[str, int] = "auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/
        numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigma : float = 2.0
        Sets the thresholds for classifing a signal window inside a state:
        the window is contained in the state if it is entirely contained
        inside number_of_sigma * state.sigms times from state.mean.

    Returns
    -------
    state_list : List[StateUni]
        List of the identified states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    time_res_analysis : np.ndarray of shape (len(tau_window_list), 2)
        For each analyzed value in `tau_window_list`, it contains
        the number of clusters identified and the fraction of unclassified
        data points.

    pop_list : List[List[float]]
        For each analyzed value in `tau_window_list`, it contains
        the fraction of data points contained in each state. So pop_list[i][j]
        is the fraction of data points classified in the j-th state using the
        i-th value of tau_window_list.

    Notes
    -----
    The complete results of the clustering (the labels and the list of states)
    is given for the selected value of `tau_window`.
    Only the number of states and the fraction of unclassified data points are
    returned for the list of time resolutions analysed. This is a tool to
    help inform the choice of `tau_window`.

    References
    ----------
    https://arxiv.org/abs/2402.07786

    Examples
    --------

    >>> from sklearn.something import onion_uni
    >>> matrix = array_with_timeseries_data
    >>> state_list, labels = onion_uni(matrix, tau_window=10)
    """

    est = OnionUni(
        tau_window=tau_window,
        tau_window_list=tau_window_list,
        bins=bins,
        number_of_sigmas=number_of_sigmas,
    )
    est.fit(matrix)

    return est.state_list_, est.labels_, est.time_res_analysis_, est.pop_list_


class OnionUni(BaseEstimator, ClusterMixin):
    """Perform onion clustering from data array.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering, corresponding to the length
        of the windows in which the time-series are segmented.

    tau_window_list : List[int]
        The list of time resolutions at which the fast analysis will
        be performed. If None (default), use a logspaced list between 2 and
        the entire trajectory length.

    bins: Union[str, int] = "auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/
        numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigma : float = 2.0
        Sets the thresholds for classifing a signal window inside a state:
        the window is contained in the state if it is entirely contained
        inside number_of_sigma * state.sigms times from state.mean.

    Attributes
    ----------
    state_list_ : List[StateUni]
        List of the identified states.

    labels_ : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    time_res_analysis_ : np.ndarray of shape (len(tau_window_list), 2)
        For each analyzed value in `tau_window_list`, it contains
        the number of clusters identified and the fraction of unclassified
        data points.

    pop_list_ : List[List[float]]
        For each analyzed value in `tau_window_list`, it contains
        the fraction of data points contained in each state. So pop_list[i][j]
        is the fraction of data points classified in the j-th state using the
        i-th value of tau_window_list.

    Notes
    -----
    The complete results of the clustering (the labels and the list of states)
    is given for the selected value of `tau_window`.
    Only the number of states and the fraction of unclassified data points are
    returned for the list of time resolutions analysed. This is a tool to
    help inform the choice of `tau_window`.

    References
    ----------
    https://arxiv.org/abs/2402.07786

    Examples
    --------

    >>> from sklearn.something import OnionUni
    >>> matrix = array_with_timeseries_data
    >>> clustering = OnionUni(matrix, tau_window=10).fit(matrix)
    """

    def __init__(
        self,
        tau_window: int = 2,
        tau_window_list=None,
        bins: Union[str, int] = "auto",
        number_of_sigmas: float = 2.0,
    ):
        self.tau_window = tau_window
        self.tau_window_list = tau_window_list
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas

    def fit(self, X, y=None):
        """Perform onion clustering from data array.

        Parameters
        ----------
        X : ndarray of shape (n_particles, n_frames)
            The values of the signal for each particle at each frame.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        X = self._validate_data(X, accept_sparse=False)

        if X.ndim != 2:
            raise ValueError("Expected 2-dimensional input data.")

        if X.shape[0] <= 1:
            raise ValueError("n_samples = 1")

        if X.shape[1] <= 1:
            raise ValueError("n_features = 1")

        # Check for complex input
        if not (
            np.issubdtype(X.dtype, np.floating)
            or np.issubdtype(X.dtype, np.integer)
        ):
            raise ValueError("Complex data not supported")

        X = X.copy()  # copy to avoid in-place modification

        cl_ob = onion_inner(
            X,
            self.tau_window,
            self.tau_window_list,
            self.bins,
            self.number_of_sigmas,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels
        self.time_res_analysis_ = np.array(
            [
                cl_ob.tau_window_list,
                cl_ob.number_of_states,
                cl_ob.fraction_0,
            ]
        ).T
        self.pop_list_ = cl_ob.list_of_pop

        return self

    def fit_predict(self, X, y=None):
        """Compute clusters from a data matrix and predict labels.

        Parameters
        ----------
        X : ndarray of shape (n_particles, n_frames)
            The values of the signal for each particle at each frame.

        Returns
        -------
        labels_ : ndarray of shape (n_samples,)
            Cluster labels. Unclassified points are given the label 0.
        """
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {
            "tau_window": self.tau_window,
            "tau_window_list": self.tau_window_list,
            "bins": self.bins,
            "number_of_sigmas": self.number_of_sigmas,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
