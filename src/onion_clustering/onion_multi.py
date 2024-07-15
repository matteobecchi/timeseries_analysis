"""onion-clustering for multivariate time-series."""

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from onion_clustering._internal.main_2d import main as onion_inner


def onion_multi(
    matrix: np.ndarray,
    n_windows: int = 2,
    n_dims: int = 2,
    bins: Union[str, int] = "auto",
    number_of_sigmas: float = 2.0,
):
    """Perform onion clustering from data array.

    Parameters
    ----------
    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    n_windows : int
        The number of windows in which the signal is divided for the analysis.

    n_dims : int = 2
        Number of components. Must be 2 or 3.

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

    References
    ----------
    https://arxiv.org/abs/2402.07786

    Examples
    --------

    >>> from sklearn.something import onion_uni
    >>> matrix = array_with_timeseries_data
    >>> state_list, labels = onion_multi(matrix)
    """

    est = OnionMulti(
        n_windows=n_windows,
        bins=bins,
        number_of_sigmas=number_of_sigmas,
    )
    est.fit(matrix)

    return est.state_list_, est.labels_


class OnionMulti(BaseEstimator, ClusterMixin):
    """Perform onion clustering from data array.

    Parameters
    ----------
    n_windows : int
        The number of windows in which the signal is divided for the analysis.

    n_dims : int = 2
        Number of components. Must be 2 or 3.

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

    References
    ----------
    https://arxiv.org/abs/2402.07786

    Examples
    --------

    >>> from sklearn.something import OnionMulti
    >>> X = array_with_timeseries_data
    >>> clustering = OnionMulti().fit(X)
    """

    def __init__(
        self,
        n_windows: int = 2,
        ndims: int = 2,
        bins: Union[str, int] = "auto",
        number_of_sigmas: float = 2.0,
    ):
        self.n_windows = n_windows
        self.bins = bins
        self.ndims = ndims
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

        # Check compatibility of array shapes
        n_particles = int(X.shape[0] / self.n_windows)
        if X.shape[0] > n_particles * self.n_windows:
            X = X[: n_particles * self.n_windows]

        tau_window = int(X.shape[1] / self.ndims)
        if X.shape[1] > tau_window * self.ndims:
            X = X[:, : tau_window * self.ndims]

        cl_ob = onion_inner(
            X,
            self.n_windows,
            self.ndims,
            self.bins,
            self.number_of_sigmas,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels

        return self

    def fit_predict(self, X, y=None):
        """Compute clusters from a data matrix and predict labels.

        Parameters
        ----------
        X : ndarray of shape (n_particles, n_frames)
            The values of the signal for each particle at each frame.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Unclassified points are given the label 0.
        """
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {
            "n_windows": self.n_windows,
            "ndims": self.ndims,
            "bins": self.bins,
            "number_of_sigmas": self.number_of_sigmas,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
