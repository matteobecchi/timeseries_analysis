"""
Code for clustering of multivariate (2- or 3-dimensional) time-series data.
See the documentation for all the details.
"""

from typing import List, Tuple, Union

import numpy as np
from onion_clustering._internal.classes import ClusteringObject2D
from onion_clustering._internal.first_classes import (
    MultiData,
    Parameters,
    StateMulti,
)
from onion_clustering._internal.functions import (
    custom_fit,
    find_half_height_around_max,
    find_minima_around_max,
    moving_average_2d,
    relabel_states_2d,
)

OUTPUT_FILE = "onion_clustering_log.txt"


def all_the_input_stuff(
    matrix: np.ndarray,
    ndims: int,
    bins: Union[int, str],
    number_of_sigmas: float,
) -> ClusteringObject2D:
    """
    Data preprocessing for the analysis.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles * n_windows, tau_window * dims)
        The values of the signal for each particle at each frame.

    n_windows : int
        Number of time windows in the simulation.

    ndims : int
        Number of components. Must be 2 or 3.

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

    ClusteringObject2D

    Notes
    -----

    - Reads analysis parameters
    - Reads input raw data
    - Creates and returns the ClusteringObject2D for the analysis
    """
    tau_window = int(matrix.shape[1] / ndims)
    reshaped_matrix = np.reshape(matrix, (ndims, -1, tau_window))

    par = Parameters(tau_window, bins, number_of_sigmas)
    data = MultiData(reshaped_matrix)
    clustering_object = ClusteringObject2D(par, data)

    return clustering_object


def gauss_fit_max(
    m_clean: np.ndarray,
    m_limits: np.ndarray,
    bins: Union[int, str],
    par: Parameters,
) -> Union[StateMulti, None]:
    """
    Selection of the optimal region and parameters in order to fit a state.

    Parameters
    ----------

    m_clean : ndarray
        The data points.

    m_limits : ndarray
        The min and max of the data points.

    bins : Union[int, str]
        The histogram binning rule.

    Returns
    -------

    state : StateMulti
        Object containing Gaussian fit parameters (mu, sigma, area),
        or None if the fit fails.

    Notes
    -----

    - Computes the data histogram
    - If the bins edges are longer than 40, smooths the histogram with gap = 3
    - Finds the maximum
    - Finds the interval between the two surriunding minima
    - Tries to perform the Gaussian fit in it
    - Finds the interval between the two half heigth points
    - Tries to perform the Gaussian fit in it
    - Compares the two fits and choose the one with higher goodness
    - Create the State object
    - Prints State's information to file
    """
    flat_m = m_clean.reshape(
        (m_clean.shape[0] * m_clean.shape[1], m_clean.shape[2])
    )
    if bins == "auto":
        bins = max(int(np.power(m_clean.size, 1 / 3) * 2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)

    gap = 1
    edges_sides = np.array([e.size for e in edges])
    if np.all(edges_sides > 49):
        gap = int(np.min(edges_sides) * 0.02) * 2
        if gap % 2 == 0:
            gap += 1

    counts = moving_average_2d(counts, gap)

    def find_max_index(data: np.ndarray):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    max_ind = find_max_index(counts)

    minima = find_minima_around_max(counts, max_ind, gap)

    popt_min: List[float] = []
    cumulative_r2_min = 0.0
    for dim in range(m_clean.shape[2]):
        try:
            flag_min, r_2, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_min.extend(popt)
            cumulative_r2_min += r_2
        except RuntimeError:
            popt_min = []
            flag_min = False

    minima = find_half_height_around_max(counts, max_ind, gap)

    popt_half: List[float] = []
    cumulative_r2_half = 0.0
    for dim in range(m_clean.shape[2]):
        try:
            flag_half, r_2, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_half.extend(popt)
            cumulative_r2_half += r_2
        except RuntimeError:
            popt_half = []
            flag_half = False

    r_2 = cumulative_r2_min
    if flag_min == 1 and flag_half == 0:
        popt = np.array(popt_min)
    elif flag_min == 0 and flag_half == 1:
        popt = np.array(popt_half)
        r_2 = cumulative_r2_half
    elif flag_min * flag_half == 1:
        if cumulative_r2_min >= cumulative_r2_half:
            popt = np.array(popt_min)
        else:
            popt = np.array(popt_half)
            r_2 = cumulative_r2_half
    else:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\tWARNING: this fit is not converging.", file=dump)
        return None
    if len(popt) != m_clean.shape[2] * 3:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\tWARNING: this fit is not converging.", file=dump)
        return None

    mean, sigma, area = [], [], []
    for dim in range(m_clean.shape[2]):
        mean.append(popt[3 * dim])
        sigma.append(popt[3 * dim + 1])
        area.append(popt[3 * dim + 2])
    state = StateMulti(np.array(mean), np.array(sigma), np.array(area))
    state.build_boundaries(par.number_of_sigmas)

    if m_clean.shape[2] == 2:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}],"
                f" sigma = [{popt[1]:.4f}, {popt[4]:.4f}],"
                f" area = {popt[2]:.4f}, {popt[5]:.4f}",
                file=dump,
            )
            print(f"\tFit coeff r^2 = {r_2}", file=dump)
    elif m_clean.shape[2] == 3:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], "
                f"sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], "
                f"area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}",
                file=dump,
            )
            print(f"\tFit coeff r^2 = {r_2}", file=dump)

    return state


def find_stable_trj(
    cl_ob: ClusteringObject2D,
    state: StateMulti,
    tmp_labels: np.ndarray,
    lim: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Identification of windows contained in a certain state.

    Parameters
    ----------

    cl_ob : ClusteringObject2D
        The clustering object.

    state : StateMulti
        The state.

    tmp_labels : ndarray of shape (n_particles, n_windows)
        Contains the cluster labels of all the signal windows.

    lim : int
        The algorithm iteration.

    Returns
    -------

    m2_array : ndarray
        Array of still unclassified data points.

    window_fraction : float
        Fraction of windows classified in this state.

    env_0 : bool
        Indicates if there are still unclassified data points.

    Notes
    -----

    - Initializes some useful variables
    - Selects the data windows contained inside the state
    - Updates tmp_labels with the newly classified data windows
    - Calculates the fraction of stable windows found and prints it
    - Creates a np.ndarray to store still unclassified windows
    - Sets the value of env_0 to signal still unclassified data points
    """

    m_clean = cl_ob.data.matrix

    mask_unclassified = tmp_labels < 0.5
    shifted = m_clean - state.mean
    rescaled = shifted / state.axis
    squared_distances = np.sum(rescaled**2, axis=2)
    mask_dist = np.max(squared_distances, axis=1) <= 1.0
    mask = mask_unclassified & mask_dist

    tmp_labels[mask] = lim + 1
    counter = np.sum(mask)

    mask_remaining = mask_unclassified & ~mask
    remaning_data = m_clean[mask_remaining]
    m2_array = np.array(remaning_data)

    if tmp_labels.size == 0:
        return m2_array, 0.0, False
    else:
        window_fraction = counter / tmp_labels.size

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"\tFraction of windows in state {lim + 1}"
            f" = {window_fraction:.3}",
            file=dump,
        )

    env_0 = True
    if len(m2_array) == 0:
        env_0 = False

    return m2_array, window_fraction, env_0


def iterative_search(
    cl_ob: ClusteringObject2D,
) -> Tuple[ClusteringObject2D, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Parameters
    ----------

    cl_ob : ClusteringObject2D
        The clustering object.

    Returns
    -------

    cl_ob : ClusteringObject2D
        Updated with the clustering results.

    env_0 : bool
        Indicates if there are unclassified data points.

    Notes
    -----

    - Initializes some useful variables
    - At each ieration:
        - performs the Gaussian fit and identifies the new proposed state
        - if no state is identified, break
        - finds the windows contained inside the proposed state
        - if no data points are remaining, break
        - otherwise, repeats
    - Updates the clusering object with the number of iterations
    - Calls "relable_states_2d" to sort and clean the state list, and updates
        the clustering object
    """
    bins = cl_ob.par.bins
    tmp_labels = np.zeros((cl_ob.data.matrix.shape[0],)).astype(int)

    states_list = []
    m_copy = cl_ob.data.matrix
    iteration_id = 1
    states_counter = 0
    env_0 = False
    while True:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(f"- Iteration {iteration_id - 1}", file=dump)
        state = gauss_fit_max(
            m_copy, np.array(cl_ob.data.range), bins, cl_ob.par
        )

        if state is None:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                print(
                    "- Iterations interrupted because fit does not converge.",
                    file=dump,
                )
            break

        m_new, counter, env_0 = find_stable_trj(
            cl_ob, state, tmp_labels, states_counter
        )

        state.perc = counter
        if counter > 0.0:
            states_list.append(state)
        states_counter += 1
        iteration_id += 1
        if counter <= 0.0:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                print(
                    "- Iterations interrupted because last state is empty.",
                    file=dump,
                )
            break
        if m_new.size == 0:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                print(
                    "- Iterations interrupted because all data "
                    "points assigned.",
                    file=dump,
                )
            break
        m_copy = m_new

    cl_ob.iterations = len(states_list)

    all_the_labels, list_of_states = relabel_states_2d(tmp_labels, states_list)
    cl_ob.data.labels = np.reshape(all_the_labels, (all_the_labels.size,)) - 1
    cl_ob.state_list = list_of_states

    return cl_ob, env_0


def full_output_analysis(cl_ob: ClusteringObject2D):
    """
    The complete clustering analysis with the input parameters.

    Parameters
    - cl_ob (ClusteringObject2D): the clustering object
    - full_out (bool): activates the full output printing

    Returns:
    - cl_ob (ClusteringObject2D): the upodated clustering object,
    with the clustering resutls

    - Preprocesses the data
    - Performs the clustering with the iterative search and classification
    - If no classification is found, return

    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"* Complete analysis, tau_window = {cl_ob.par.tau_w}\n", file=dump
        )

    cl_ob, _ = iterative_search(cl_ob)

    if len(cl_ob.state_list) == 0:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("* No possible classification was found.", file=dump)


def main(
    matrix: np.ndarray,
    ndims: int,
    bins: Union[int, str],
    number_of_sigmas: float,
) -> ClusteringObject2D:
    """
    Returns the clustering object with the analysis.

    Parameters
    ----------
    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    n_dims : int
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

    clustering_object : ClusteringObject2D
        The final clustering object.

    Notes
    -----

    - Reads the data and the parameters
    - Performs the quick analysis for all the values in tau_window_list
    - Performs a detailed analysis with the selected parameters
    """
    clustering_object = all_the_input_stuff(
        matrix,
        ndims,
        bins,
        number_of_sigmas,
    )

    full_output_analysis(clustering_object)

    return clustering_object
