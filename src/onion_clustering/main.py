"""
Code for clustering of univariate time-series data.
See the documentation for all the details.
"""

import copy
from typing import List, Tuple, Union

import numpy as np
import scipy.signal

from onion_clustering.classes import ClusteringObject1D
from onion_clustering.first_classes import Parameters, StateUni, UniData
from onion_clustering.functions import (
    gaussian,
    moving_average,
    param_grid,
    relabel_states,
    set_final_states,
)

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = "onion_clustering_log.txt"


def all_the_input_stuff(
    matrix,
    tau_window,
    tau_window_list,
    bins,
) -> ClusteringObject1D:
    """
    Data preprocessing for the analysis.

    Returns:
        ClusteringObject1D

    - Reads analysis parameters
    - Reads input raw data
    - Removes initial 't_delay' frames
    - Creates and returns the ClusteringObject1D for the analysis
    """
    par = Parameters(tau_window, tau_window_list, bins)
    data = UniData(matrix)
    clustering_object = ClusteringObject1D(par, data)

    return clustering_object


def perform_gauss_fit(
    param: List[int], data: List[np.ndarray], int_type: str
) -> Tuple[bool, int, np.ndarray]:
    """
    Gaussian fit on the data histogram.

    Args:
        param (List[int]): a list of the parameters for the fit:
            initial index,
            final index,
            index of the max,
            amount of data points,
            gap value for histogram smoothing
        data (List[np.ndarray]): a list of the data for the fit
            histogram binning,
            histogram counts
        int_type (str): the type of the fitting interval ('max' or 'half')

    Returns:
        A boolean value for the fit convergence
        goodness (int): the fit quality (max is 5)
        popt (np.ndarray): the optimal gaussians fit parameters

    - Trys to perform the fit with the specified parameters
    - Computes the fit quality by checking if some requirements are satisfied
    - If the fit fails, returns (False, 5, np.empty(3))
    """
    id0, id1, max_ind, n_data, gap = param
    bins, counts = data

    goodness = 5
    selected_bins = bins[id0:id1]
    selected_counts = counts[id0:id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[id0] - bins[id1]) / 6
    area0 = counts[max_ind] * np.sqrt(np.pi) * sigma0
    try:
        popt, pcov, _, _, _ = scipy.optimize.curve_fit(
            gaussian,
            selected_bins,
            selected_counts,
            p0=[mu0, sigma0, area0],
            full_output=True,
        )
        if popt[1] < 0:
            popt[1] = -popt[1]
            popt[2] = -popt[2]
        gauss_max = popt[2] * np.sqrt(np.pi) * popt[1]
        if gauss_max < area0 / 2:
            goodness -= 1
        popt[2] *= n_data
        if popt[0] < selected_bins[0] or popt[0] > selected_bins[-1]:
            goodness -= 1
        if popt[1] > selected_bins[-1] - selected_bins[0]:
            goodness -= 1
        perr = np.sqrt(np.diag(pcov))
        for j, par_err in enumerate(perr):
            if par_err / popt[j] > 0.5:
                goodness -= 1
        if id1 - id0 <= gap:
            goodness -= 1
        return True, goodness, popt
    except RuntimeError:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\t" + int_type + " fit: Runtime error. ", file=dump)
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )
    except TypeError:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\t" + int_type + " fit: TypeError.", file=dump)
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )
    except ValueError:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\t" + int_type + " fit: ValueError.", file=dump)
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )


def gauss_fit_max(
    m_clean: np.ndarray, par: Parameters
) -> Union[StateUni, None]:
    """
    Selection of the optimal interval and parameters in order to fit a state.

    Args:
        m_clean (np.ndarray): the data points
        par (Parameters): object containing parameters for the analysis.

    Returns:
        state (StateUni): object containing Gaussian fit parameters
            (mu, sigma, area), or None if the fit fails.

    - Computes the data histogram
    - If the bins are more than 50, smooths the histogram with gap = 3
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
    flat_m = m_clean.flatten()
    counts, bins = np.histogram(flat_m, bins=par.bins, density=True)

    gap = 1
    if bins.size > 50:
        gap = 3
    counts = moving_average(counts, gap)
    bins = moving_average(bins, gap)
    if (counts == 0.0).any():
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                "\tWARNING: there are empty bins. "
                "Consider reducing the number of bins.",
                file=dump,
            )

    max_val = counts.max()
    max_ind = counts.argmax()

    min_id0 = np.max([max_ind - gap, 0])
    min_id1 = np.min([max_ind + gap, counts.size - 1])
    while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
        min_id0 -= 1
    while min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]:
        min_id1 += 1

    fit_param = [min_id0, min_id1, max_ind, flat_m.size, gap]
    fit_data = [bins, counts]
    flag_min, goodness_min, popt_min = perform_gauss_fit(
        fit_param, fit_data, "Min"
    )

    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val / 2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val / 2:
        half_id1 += 1

    fit_param = [half_id0, half_id1, max_ind, flat_m.size, gap]
    fit_data = [bins, counts]
    flag_half, goodness_half, popt_half = perform_gauss_fit(
        fit_param, fit_data, "Half"
    )

    goodness = goodness_min
    if flag_min == 1 and flag_half == 0:
        popt = popt_min
    elif flag_min == 0 and flag_half == 1:
        popt = popt_half
        goodness = goodness_half
    elif flag_min * flag_half == 1:
        if goodness_min >= goodness_half:
            popt = popt_min
        else:
            popt = popt_half
            goodness = goodness_half
    else:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\tWARNING: this fit is not converging.", file=dump)
        return None

    state = StateUni(popt[0], popt[1], popt[2])
    state.build_boundaries(NUMBER_OF_SIGMAS)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"\tmu = {state.mean:.4f}, sigma = {state.sigma:.4f},"
            f" area = {state.area:.4f}",
            file=dump,
        )
        print("\tFit goodness = " + str(goodness), file=dump)

    return state


def find_stable_trj(
    cl_ob: ClusteringObject1D,
    state: StateUni,
    tmp_labels: np.ndarray,
    lim: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Identification of windows contained in a certain state.

    Args:
        cl_ob (ClusteringObject1D): the clustering object
        state (StateUni): the state
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
        m_clean.shape[0], number_of_windows, tau_window
    )
    mask_inf = np.min(m_reshaped, axis=2) >= state.th_inf[0]
    mask_sup = np.max(m_reshaped, axis=2) <= state.th_sup[0]
    mask = mask_unclassified & mask_inf & mask_sup

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
    m2_array = np.array(remaning_data)

    env_0 = True
    if len(m2_array) == 0:
        env_0 = False

    return m2_array, window_fraction, env_0


def iterative_search(
    cl_ob: ClusteringObject1D,
) -> Tuple[ClusteringObject1D, np.ndarray, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Args:
        cl_ob (ClusteringObject1D): the clustering object

    Returns:
        cl_ob (ClusteringObject1D): updated with the clustering results
        atl (np.ndarray): temporary array of labels
        env_0 (bool): indicates if there are unclassified data points

    - Initializes some useful variables
    - At each ieration:
        - performs the Gaussian fit and identifies the new proposed state
        - if no state is identified, break
        - finds the windows contained inside the proposed state
        - if no data points are remaining, break
        - otherwise, repeats
    - Updates the clusering object with the number of iterations
    - Calls "relable_states" to sort and clean the state list, and updates
    the clustering object
    """
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
            print(f"_ Iteration {iteration_id - 1}", file=dump)
        state = gauss_fit_max(m_copy, cl_ob.par)

        if state is None:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                print(
                    "_ Iterations interrupted because "
                    "fit does not converge.",
                    file=dump,
                )
            break

        m_next, counter, env_0 = find_stable_trj(
            cl_ob, state, tmp_labels, states_counter
        )

        state.perc = counter
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

        m_copy = m_next

    cl_ob.iterations = len(states_list)

    atl, lis = relabel_states(tmp_labels, states_list)
    cl_ob.state_list = lis

    return cl_ob, atl, env_0


def timeseries_analysis(
    cl_ob: ClusteringObject1D, tau_w: int
) -> Tuple[int, float]:
    """
    The clustering analysis to compute the dependence on time resolution.

    Args:
        cl_ob (ClusteringObject1D): the clustering object
        tau_w (int): the time resolution for the analysis

    Returns:
        num_states (int): number of identified states
        fraction_0 (float): fraction of unclassified data points

    - Creates a copy of the clustering object and of the parameters
    - Preprocesses the data with the selected parameters
    - Performs the clustering with the iterative search and classification
    - If no classification is found, cleans the memory and return
    - Otherwise, final states are identified by "set_final_states"
    - Number of states and fraction of unclassified points are computed
    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(f"* tau_window = {tau_w}", file=dump)

    tmp_cl_ob = copy.deepcopy(cl_ob)
    tmp_cl_ob.par.tau_w = tau_w

    tmp_cl_ob, tmp_labels, env_0 = iterative_search(tmp_cl_ob)

    if len(tmp_cl_ob.state_list) == 0:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("* No possible classification was found.", file=dump)
        del tmp_cl_ob
        return 1, 1.0

    tmp_cl_ob.state_list, tmp_cl_ob.data.labels = set_final_states(
        tmp_cl_ob.state_list, tmp_labels, tmp_cl_ob.data.range
    )

    fraction_0 = 1 - np.sum([state.perc for state in tmp_cl_ob.state_list])
    n_states = len(tmp_cl_ob.state_list)
    if env_0:
        n_states += 1
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"* Number of states identified: {n_states}, [{fraction_0}]\n",
            file=dump,
        )

    del tmp_cl_ob
    return n_states, fraction_0


def full_output_analysis(cl_ob: ClusteringObject1D):
    """
    The complete clustering analysis with the input parameters.

    Args:
        cl_ob (ClusteringObject1D): the clustering object

    - Preprocesses the data
    - Performs the clustering with the iterative search and classification
    - If no classification is found, return
    - Otherwise, final states are identified by "set_final_states"
    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"* Complete analysis, tau_window = {cl_ob.par.tau_w}\n", file=dump
        )
    cl_ob, tmp_labels, _ = iterative_search(cl_ob)

    if len(cl_ob.state_list) == 0:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("* No possible classification was found.", file=dump)

    cl_ob.state_list, cl_ob.data.labels = set_final_states(
        cl_ob.state_list, tmp_labels, cl_ob.data.range
    )
    cl_ob.data.labels = cl_ob.create_all_the_labels()


def time_resolution_analysis(cl_ob: ClusteringObject1D):
    """
    Explore parameter space and compute the dependence on time resolution.

    Args:
        cl_ob (ClusteringObject1D): the clustering object

    - Generates the parameters' grid
    - Performs and stores the clustering for all the parameters' combinations
    - Prints the output to file
    - Updates the clustering object with the analysis results
    """
    if cl_ob.par.tau_w_list is None:
        tau_window_list = param_grid(cl_ob.data.num_of_steps)
    else:
        tau_window_list = cl_ob.par.tau_w_list

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
    tau_window_list,
    bins,
) -> ClusteringObject1D:
    """
    Returns the clustering object with the analysis.

    Parameters
    ----------
    To write

    Returns
    -------
    clustering_object (ClusteringObject1D): the final clustering object

    Notes
    -----
    - Reads the data and the parameters
    - Explore the parameters (tau_window, t_smooth) space
    - Performs a detailed analysis with the selected parameters
    """
    print("##############################################################")
    print("# If you publish results using onion-clustering, please cite #")
    print("# this work: https://doi.org/10.48550/arXiv.2402.07786.      #")
    print("##############################################################")

    clustering_object = all_the_input_stuff(
        matrix, tau_window, tau_window_list, bins
    )

    time_resolution_analysis(clustering_object)

    full_output_analysis(clustering_object)

    return clustering_object
