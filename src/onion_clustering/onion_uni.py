"""onion-clustering for univariate time-series."""

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from onion_clustering._internal.main import main as onion_inner


def onion_uni(
    matrix: np.ndarray,
    bins: Union[str, int] = "auto",
    number_of_sigmas: float = 2.0,
):
    """Perform onion clustering from data array.

    References:
        https://www.pnas.org/doi/abs/10.1073/pnas.2403771121

    Parameters:
        matrix:
            Shape (n_particles * n_windows, tau_window). The values of the signal for each particle at each frame.

        bins (default = "auto"):
            The number of bins used for the construction of the histograms.
            Can be an integer value, or "auto".
            If "auto", the default of numpy.histogram_bin_edges is used
            (see https://numpy.org/doc/stable/reference/generated/
            numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

        number_of_sigmas (default = 2.0):
            Sets the thresholds for classifing a signal window inside a state:
            the window is contained in the state if it is entirely contained
            inside number_of_sigma * state.sigms times from state.mean.

    Returns:
        - List of the identified states.
        - np.ndarray of shape (n_particles, n_frames). Cluster labels for each point. Unclassified points are given the label -1.
    """

    est = OnionUni(
        bins=bins,
        number_of_sigmas=number_of_sigmas,
    )
    est.fit(matrix)

    return est.state_list_, est.labels_


class OnionUni(BaseEstimator, ClusterMixin):
    """Perform onion clustering from data array.

    References:
        https://www.pnas.org/doi/abs/10.1073/pnas.2403771121

    Parameters:
        bins (default = "auto"):
            The number of bins used for the construction of the histograms. Can be an integer value, or "auto".
            If "auto", the default of numpy.histogram_bin_edges is used
            (see https://numpy.org/doc/stable/reference/generated/
            numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

        number_of_sigma (default = 2.0):
            Sets the thresholds for classifing a signal window inside a state:
            the window is contained in the state if it is entirely contained
            inside number_of_sigma * state.sigms times from state.mean.

    Attributes:
        state_list_:
            List of the identified states.

        labels_:
            np.ndarray of shape (n_particles, n_frames). Cluster labels
            for each point. Unclassified points are given the label -1.
    """

    def __init__(
        self,
        bins: Union[str, int] = "auto",
        number_of_sigmas: float = 2.0,
    ):
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas

    def fit(self, X, y=None):
        """Perform onion clustering from data array.

        Parameters:
            X:
                np.ndarray of shape (n_particles * n_windows, tau_window).
                The values of the signal for each particle at each frame.

        Returns:
            A fitted instance of self.
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
            self.bins,
            self.number_of_sigmas,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels

        return self

    def fit_predict(self, X, y=None):
        """Compute clusters from a data matrix and predict labels.

        Parameters:
            X:
                np.ndarray of shape (n_particles * n_windows, tau_window).
                The values of the signal for each particle at each frame.

        Returns:
            np.ndarray of shape (n_particles * n_windows,). Cluster labels. Unclassified points are given the label -1.
        """
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {
            "bins": self.bins,
            "number_of_sigmas": self.number_of_sigmas,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
