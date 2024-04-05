"""onion-clustering for univariate time-series."""

from typing import List, Union

import numpy as np

from onion_clustering.main import main as onion_inner


def onion_uni(
    X,
    tau_window,
    tau_window_list: Union[List[int], None] = None,
    bins: Union[str, int] = "auto",
):
    """Perform onion clustering from data array.

    Parameters
    ----------
    X : ndarray of shape (n_particles, n_frames)
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
    >>> X = array_with_timeseries_data
    >>> state_list, labels = onion_uni(X, tau_window=10)
    """

    est = OnionUni(
        tau_window=tau_window,
        tau_window_list=tau_window_list,
        bins=bins,
    )
    est.fit(X)

    return est.state_list_, est.labels_, est.time_res_analysis_


class OnionUni:
    """Perform onion clustering from data array.

    Parameters
    ----------
    X : ndarray of shape (n_particles, n_frames)
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

    time_res_analysis_ : np.ndarray of shape (len(tau_window_list), 2)
        For each analyzed value in `tau_window_list`, it contains
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
    >>> X = array_with_timeseries_data
    >>> clustering = OnionUni(X, tau_window=10).fit(X)
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

    def fit(self, X):
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
        cl_ob = onion_inner(
            X,
            self.tau_window,
            self.tau_window_list,
            self.bins,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels
        self.time_res_analysis_ = np.array(
            [cl_ob.tau_window_list, cl_ob.number_of_states, cl_ob.fraction_0]
        ).T

    def fit_predict(self, X):
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
        self.fit(X)
        return self.labels_
