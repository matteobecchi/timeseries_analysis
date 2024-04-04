"""
Code for clustering of multivariate (2- or 3-dimensional) time-series data.
See the documentation for all the details.
"""

import copy
from typing import List, Tuple, Union

import numpy as np

from onion_clustering.classes import ClusteringObject2D
from onion_clustering.first_classes import MultiData, Parameters, StateMulti
from onion_clustering.functions import (
    custom_fit,
    find_half_height_around_max,
    find_minima_around_max,
    moving_average_2d,
    param_grid,
    relabel_states_2d,
)

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = "onion_clustering_log.txt"


def all_the_input_stuff(
    matrix,
    tau_window,
    bins,
    num_tau_w,
    min_tau_w,
    max_tau_w,
) -> ClusteringObject2D:
    """
    Data preprocessing for the analysis.

    Returns:
        ClusteringObject2D

    - Reads analysis parameters
    - Reads input raw data
    - Removes initial 't_delay' frames
    - Creates and returns the ClusteringObject2D for the analysis
    """
    par = Parameters(tau_window, bins, num_tau_w, min_tau_w, max_tau_w)
    data = MultiData(matrix)
    clustering_object = ClusteringObject2D(par, data)

    return clustering_object


def gauss_fit_max(
    m_clean: np.ndarray,
    m_limits: np.ndarray,
    bins: Union[int, str],
) -> Union[StateMulti, None]:
    """
    Selection of the optimal region and parameters in order to fit a state.

    Args:
        m_clean (np.ndarray): the data points
        m_limits (np.ndarray): the min and max of the data points
        bins (Union[int, str]): the histogram binning
        filename (str): name of the output plot file
        full_out (bool): activates the full output printing

    Returns:
        state (StateMulti): object containing Gaussian fit parameters
            (mu, sigma, area), or None if the fit fails.

    - Computes the data histogram
    - If the bins edges are longer than 40, smooths the histogram with gap = 3
    - Finds the maximum
    - Finds the interval between the two surriunding minima
    - Tries to perform the Gaussian fit in it
    - Finds the interval between the two half heigth points
    - Tries to perform the Gaussian fit in it
    - Compares the two fits and choose the one with higher goodness
    - Create the State object
    - Prints State's information
    - Plots the histogram with the best fit
    """
    flat_m = m_clean.reshape(
        (m_clean.shape[0] * m_clean.shape[1], m_clean.shape[2])
    )
    if bins == "auto":
        bins = max(int(np.power(m_clean.size, 1 / 3) * 2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)

    gap = 1
    if np.all([e.size > 40 for e in edges]):
        gap = 3
    counts = moving_average_2d(counts, gap)

    def find_max_index(data: np.ndarray):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    max_ind = find_max_index(counts)

    minima = find_minima_around_max(counts, max_ind, gap)

    popt_min: List[float] = []
    goodness_min = 0
    for dim in range(m_clean.shape[2]):
        try:
            flag_min, goodness, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, gap, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_min.extend(popt)
            goodness_min += goodness
        except RuntimeError:
            popt_min = []
            flag_min = False
            goodness_min -= 5

    minima = find_half_height_around_max(counts, max_ind, gap)

    popt_half: List[float] = []
    goodness_half = 0
    for dim in range(m_clean.shape[2]):
        try:
            flag_half, goodness, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, gap, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_half.extend(popt)
            goodness_half += goodness
        except RuntimeError:
            popt_half = []
            flag_half = False
            goodness_half -= 5

    goodness = goodness_min
    if flag_min == 1 and flag_half == 0:
        popt = np.array(popt_min)
    elif flag_min == 0 and flag_half == 1:
        popt = np.array(popt_half)
        goodness = goodness_half
    elif flag_min * flag_half == 1:
        if goodness_min >= goodness_half:
            popt = np.array(popt_min)
        else:
            popt = np.array(popt_half)
            goodness = goodness_half
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
    state.build_boundaries(NUMBER_OF_SIGMAS)

    if m_clean.shape[2] == 2:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}],"
                f" sigma = [{popt[1]:.4f}, {popt[4]:.4f}],"
                f" area = {popt[2]:.4f}, {popt[5]:.4f}",
                file=dump,
            )
            print("\tFit goodness = " + str(goodness), file=dump)
    elif m_clean.shape[2] == 3:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], "
                f"sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], "
                f"area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}",
                file=dump,
            )
            print("\tFit goodness = " + str(goodness), file=dump)

    return state


def find_stable_trj(
    cl_ob: ClusteringObject2D,
    state: StateMulti,
    tmp_labels: np.ndarray,
    lim: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Identification of windows contained in a certain state.

    Args:
        cl_ob (ClusteringObject2D): the clustering object
        state (StateMulti): the state
        tmp_labels (np.ndarray): contains the cluster labels of all the
            signal windows
        lim (int): the algorithm iteration

    Returns:
        m2_array (np.ndarray): array of still unclassified data points
        window_fraction (float): fraction of windows classified in this state
        env_0 (bool): indicates if there are still unclassified data points

    - Initializes some useful variables
    - Selects the data windows contained inside the state
    - Updates tmp_labels with the newly classified data windows
    - Calculates the fraction of stable windows found and prints it
    - Creates a np.ndarray to store still unclassified windows
    - Sets the value of env_0 to signal still unclassified data points

    """
    number_of_windows = tmp_labels.shape[1]
    m_clean = cl_ob.data.matrix
    tau_window = cl_ob.par.tau_w

    mask_unclassified = tmp_labels < 0.5
    m_reshaped = m_clean[:, : number_of_windows * tau_window].reshape(
        m_clean.shape[0], number_of_windows, tau_window, m_clean.shape[2]
    )
    shifted = m_reshaped - state.mean
    rescaled = shifted / state.axis
    squared_distances = np.sum(rescaled**2, axis=3)
    mask_dist = np.max(squared_distances, axis=2) <= 1.0
    mask = mask_unclassified & mask_dist

    tmp_labels[mask] = lim + 1

    counter = np.sum(mask)
    window_fraction = counter / (tmp_labels.size)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"\tFraction of windows in state {lim + 1}"
            f" = {window_fraction:.3}",
            file=dump,
        )

    remaning_data = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window * tau_window : (window + 1) * tau_window]
        remaning_data.append(r_w)
    m2_arr = np.array(remaning_data)

    env_0 = True
    if len(m2_arr) == 0:
        env_0 = False

    return m2_arr, window_fraction, env_0


def iterative_search(
    cl_ob: ClusteringObject2D,
) -> Tuple[ClusteringObject2D, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Args:
        cl_ob (ClusteringObject2D): the clustering object
        name (str): name for output figures
        full_out (bool): activates the full output printing

    Returns:
        cl_ob (ClusteringObject1D): updated with the clustering results
        env_0 (bool): indicates if there are unclassified data points

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
    num_windows = int(cl_ob.data.num_of_steps / cl_ob.par.tau_w)
    tmp_labels = np.zeros((cl_ob.data.num_of_particles, num_windows)).astype(
        int
    )

    states_list = []
    m_copy = cl_ob.data.matrix
    iteration_id = 1
    states_counter = 0
    env_0 = False
    while True:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(f"- Iteration {iteration_id - 1}", file=dump)
        state = gauss_fit_max(m_copy, np.array(cl_ob.data.range), bins)

        if state is None:
            print("* Iterations interrupted because fit does not converge. ")
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
                    "- Iterations interrupted because last state " "is empty.",
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
    cl_ob.data.labels = all_the_labels
    cl_ob.state_list = list_of_states

    return cl_ob, env_0


def timeseries_analysis(
    cl_ob: ClusteringObject2D, tau_w: int
) -> Tuple[int, float]:
    """
    The clustering analysis to compute the dependence on time resolution.

    Args:
    - cl_ob (ClusteringObject1D): the clustering object
    - tau_w (int): the time resolution for the analysis
    - t_smooth (int): the width of the moving average for the analysis
    - full_out (bool): activates the full output printing

    Returns:
    - num_states (int): number of identified states
    - fraction_0 (float): fraction of unclassified data points

    - Creates a copy of the clustering object and of the parameters
        on which the analysis will be performed
    - Preprocesses the data with the selected parameters
    - Performs the clustering with the iterative search and classification
    - If no classification is found, cleans the memory and return
    - Number of states and fraction of unclassified points are computed
    """

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(f"* tau_window = {tau_w}\n", file=dump)

    tmp_cl_ob = copy.deepcopy(cl_ob)
    tmp_cl_ob.par.tau_w = tau_w

    tmp_cl_ob, one_last_state = iterative_search(tmp_cl_ob)

    if len(tmp_cl_ob.state_list) == 0:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("* No possible classification was found.\n", file=dump)
        del tmp_cl_ob
        return 1, 1.0

    fraction_0 = 1 - np.sum([state.perc for state in tmp_cl_ob.state_list])
    n_states = len(tmp_cl_ob.state_list)
    if one_last_state:
        n_states += 1
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"- Number of states identified: {n_states}, [{fraction_0}]\n",
            file=dump,
        )

    del tmp_cl_ob
    return n_states, fraction_0


def full_output_analysis(cl_ob: ClusteringObject2D):
    """
    The complete clustering analysis with the input parameters.

    Args:
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


def time_resolution_analysis(cl_ob: ClusteringObject2D):
    """
    Explore parameter space and compute the dependence on time resolution.

    Args:
    - cl_ob (ClusteringObject1D): the clustering object
    - full_out (bool): activates the full output printing

    - Generates the parameters' grid
    - Performs and stores the clustering for all the parameters' combinations
    - Prints the output to file
    - Updates the clustering object with the analysis results
    """
    tau_window_list = param_grid(cl_ob.par, cl_ob.data.num_of_steps)
    cl_ob.tau_window_list = tau_window_list
    with open(OUTPUT_FILE, "w", encoding="utf-8") as dump:
        print("* Tau_w used:", tau_window_list, "\n", file=dump)

    number_of_states = []
    fraction_0 = []
    for tau_w in tau_window_list:
        n_s, f_0 = timeseries_analysis(cl_ob, tau_w)
        number_of_states.append(n_s)
        fraction_0.append(f_0)

    cl_ob.number_of_states = np.array(number_of_states)
    cl_ob.fraction_0 = np.array(fraction_0)


def main(
    matrix,
    tau_window,
    bins,
    num_tau_w,
    min_tau_w,
    max_tau_w,
) -> ClusteringObject2D:
    """
    Returns the clustering object with the analysis.

    Args:
    - full_output (bool): activates the full output printing

    Returns:
    - clustering_object (ClusteringObject2D): the final clustering object

    - Reads the data and the parameters
    - Explore the parameters (tau_window, t_smooth) space
    - Performs a detailed analysis with the selected parameters
    """
    print("##############################################################")
    print("# If you publish results using onion-clustering, please cite #")
    print("# this work: https://doi.org/10.48550/arXiv.2402.07786.      #")
    print("##############################################################")

    clustering_object = all_the_input_stuff(
        matrix, tau_window, bins, num_tau_w, min_tau_w, max_tau_w
    )

    time_resolution_analysis(clustering_object)

    full_output_analysis(clustering_object)

    return clustering_object
