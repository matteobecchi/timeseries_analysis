"""onion-clustering for multivariate time-series."""

from typing import List, Union

import numpy as np

from onion_clustering._internal.main_2d import main as onion_inner


def onion_multi(
    matrix,
    tau_window,
    tau_window_list: Union[List[int], None] = None,
    bins: Union[str, int] = "auto",
):
    """Perform onion clustering from data array.

    Parameters
    ----------
    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering, corresponding to the lwngth
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

    Returns
    -------
    state_list : List[StateUni]
        List of the identified states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    time_res_analysis : np.ndarray of shape (num_tau_w, 3)
        For each analyzed value of `tau_window`, it contains tau_window,
        the number of clusters identified and the fraction of unclassified
        data points.

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

    est = OnionMulti(
        tau_window=tau_window,
        tau_window_list=tau_window_list,
        bins=bins,
    )
    est.fit(matrix)

    return est.state_list_, est.labels_, est.time_res_analysis_


class OnionMulti:
    """Perform onion clustering from data array.

    Parameters
    ----------
    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering, corresponding to the lwngth
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

    Attributes
    ----------
    state_list_ : List[StateUni]
        List of the identified states.

    labels_ : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    time_res_analysis_ : np.ndarray of shape (num_tau_w, 3)
        For each analyzed value of `tau_window`, it contains tau_window,
        the number of clusters identified and the fraction of unclassified
        data points.

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
        tau_window,
        tau_window_list=None,
        bins: Union[str, int] = "auto",
    ):
        self.tau_window = tau_window
        self.tau_window_list = tau_window_list
        self.bins = bins

    def fit(self, matrix):
        """Perform onion clustering from data array.

        Parameters
        ----------
        matrix : ndarray of shape (n_particles, n_frames)
            The values of the signal for each particle at each frame.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        cl_ob = onion_inner(
            matrix,
            self.tau_window,
            self.tau_window_list,
            self.bins,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels
        self.time_res_analysis_ = np.array(
            [cl_ob.tau_window_list, cl_ob.number_of_states, cl_ob.fraction_0]
        ).T

    def fit_predict(self, matrix):
        """Compute clusters from a data matrix and predict labels.

        Parameters
        ----------
        matrix : ndarray of shape (n_particles, n_frames)
            The values of the signal for each particle at each frame.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Unclassified points are given the label 0.
        """
        self.fit(matrix)
        return self.labels_
