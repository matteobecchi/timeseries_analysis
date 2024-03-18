"""onion-clustering."""

from typing import Union

import numpy as np

from onion_clustering.main_2d import main as onion_inner


def onion_multi(
    X,
    tau_window,
    bins: Union[str, int] = "auto",
    num_tau_w=20,
    min_tau_w=2,
    max_tau_w=-1,
):
    """Perform onion clustering from data array.

    Parameters
    ----------
    X : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering.

    LIST ALL THE OTHER PARAMS

    Returns
    -------
    state_list : List[StateUni]
        List of the identified states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    Notes
    -----

    References
    ----------

    Examples
    --------

    >>> from sklearn.something import onion_uni
    >>> X = QUALCOSA
    >>> state_list, labels = onion_uni(X, tau_window=10)
    """

    est = OnionMulti(
        tau_window=tau_window,
        bins=bins,
        num_tau_w=num_tau_w,
        min_tau_w=min_tau_w,
        max_tau_w=max_tau_w,
    )
    est.fit(X)

    time_res = est.time_res_
    n_states = est.num_of_states_
    frac_0 = est.frac_0_
    time_res_analysis = np.array([time_res, n_states, frac_0]).T

    return est.state_list_, est.labels_, time_res_analysis


class OnionMulti:
    """Perform onion clustering from data array.

    Parameters
    ----------
    X : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    tau_window : int
        The time resolution for the clustering.

    LIST ALL THE OTHER PARAMS

    Attributes
    ----------
    state_list_ : List[StateUni]
        List of the identified states.

    labels_ : ndarray of shape (n_particles, n_frames)
        Cluster labels for each point. Unclassified points
        are given the label 0.

    Notes
    -----

    References
    ----------

    Examples
    --------

    >>> from sklearn.something import OnionUni
    >>> X = QUALCOSA
    >>> clustering = OnionUni(X, tau_window=10).fit(X)
    """

    def __init__(
        self,
        tau_window,
        bins: Union[str, int] = "auto",
        num_tau_w=20,
        min_tau_w=2,
        max_tau_w=-1,
    ):
        self.tau_window = tau_window
        self.bins = bins
        self.num_tau_w = num_tau_w
        self.min_tau_w = min_tau_w
        self.max_tau_w = max_tau_w

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
            self.bins,
            self.num_tau_w,
            self.min_tau_w,
            self.max_tau_w,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels
        self.time_res_ = cl_ob.tau_window_list
        self.num_of_states_ = cl_ob.number_of_states
        self.frac_0_ = cl_ob.fraction_0

    def fit_predict(self, X):
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
        self.fit(X)
        return self.labels_
