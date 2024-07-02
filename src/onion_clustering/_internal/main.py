"""
Code for clustering of univariate time-series data.
See the documentation for all the details.
"""

import copy
import warnings
from typing import List, Tuple, Union

import numpy as np
import scipy.signal
from onion_clustering._internal.classes import ClusteringObject1D
from onion_clustering._internal.first_classes import (
    Parameters,
    StateUni,
    UniData,
)
from onion_clustering._internal.functions import (
    gaussian,
    max_prob_assignment,
    param_grid,
    relabel_states,
    set_final_states,
)
from scipy.optimize import OptimizeWarning
from scipy.stats import gaussian_kde

OUTPUT_FILE = "onion_clustering_log.txt"
AREA_MAX_OVERLAP = 0.8


def all_the_input_stuff(
    matrix: np.ndarray,
    tau_window: int,
    tau_window_list: List[int],
    bins: Union[int, str],
    number_of_sigmas: float,
) -> ClusteringObject1D:
    """
    Data preprocessing for the analysis.

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

    Returns
    -------

    clustering_object : ClusteringObject1D

    Notes
    -----

    - Reads analysis parameters
    - Reads input data
    - Creates and returns the ClusteringObject1D for the analysis
    """
    par = Parameters(tau_window, tau_window_list, bins, number_of_sigmas)
    data = UniData(matrix)
    clustering_object = ClusteringObject1D(par, data)

    return clustering_object


def perform_gauss_fit(
    param: List[int], data: List[np.ndarray], int_type: str
) -> Tuple[bool, int, np.ndarray, np.ndarray]:
    """
    Gaussian fit on the data histogram.

    Parameters
    ----------

    param : List[int]
        A list of the parameters for the fit:
            initial index,
            final index,
            index of the max,
            amount of data points,
            gap value for histogram smoothing

    data : List[np.ndarray]
        A list of the data for the fit:
            histogram binning,
            histogram counts

    int_type : str
        The type of the fitting interval ('max' or 'half').

    Returns
    -------

    A boolean value for the fit convergence.

    goodness : int
        The fit quality (max is 5).

    popt : ndarray of shape (3,)
        The optimal gaussians fit parameters.

    Notes
    -----

    - Trys to perform the fit with the specified parameters
    - Computes the fit quality by checking if some requirements are satisfied
    - If the fit fails, returns (False, 5, np.empty(3))
    """
    ### Initialize return values ###
    flag = False
    coeff_det_r2 = 0
    popt = np.empty(3)
    perr = np.empty(3)

    id0, id1, max_ind, n_data = param
    bins, counts = data

    selected_bins = bins[id0:id1]
    selected_counts = counts[id0:id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[id0] - bins[id1]) / 6
    area0 = counts[max_ind] * np.sqrt(np.pi) * sigma0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                popt, pcov, infodict, _, _ = scipy.optimize.curve_fit(
                    gaussian,
                    selected_bins,
                    selected_counts,
                    p0=[mu0, sigma0, area0],
                    full_output=True,
                )
                if popt[1] < 0:
                    popt[1] = -popt[1]
                    popt[2] = -popt[2]
                popt[2] *= n_data
                perr = np.array(
                    [np.sqrt(pcov[i][i]) for i in range(popt.size)]
                )
                perr[2] *= n_data
                ss_res = np.sum(infodict["fvec"] ** 2)
                ss_tot = np.sum(
                    (selected_counts - np.mean(selected_counts)) ** 2
                )
                coeff_det_r2 = 1 - ss_res / ss_tot
                flag = True
        except OptimizeWarning:
            print(f"\t{int_type} fit: Optimize warning.", file=dump)
        except RuntimeError:
            print(f"\t{int_type} fit: Runtime error.", file=dump)
        except TypeError:
            print(f"\t{int_type} fit: TypeError.", file=dump)
        except ValueError:
            print(f"\t{int_type} fit: ValueError.", file=dump)

    return flag, coeff_det_r2, popt, perr


def gauss_fit_max(
    m_clean: np.ndarray, par: Parameters
) -> Union[StateUni, None]:
    """
    Selection of the optimal interval and parameters in order to fit a state.

    Parameters
    ----------

    m_clean : ndarray
        The data points.

    par : Parameters
        Object containing parameters for the analysis.

    Returns
    -------

    state : StateUni
        Object containing Gaussian fit parameters (mu, sigma, area),
        or None if the fit fails.

    Notes
    -----

    - Computes the data histogram
    - If the bins are more than 50, smooths the histogram with gap = 3
    - Finds the maximum
    - Finds the interval between the two surriunding minima
    - Tries to perform the Gaussian fit in it
    - Finds the interval between the two half heigth points
    - Tries to perform the Gaussian fit in it
    - Compares the two fits and choose the one with higher goodness
    - Create the State object
    - Prints State's information to file
    """
    flat_m = m_clean.flatten()

    kde = gaussian_kde(flat_m)
    if par.bins == "auto":
        bins = np.linspace(np.min(flat_m), np.max(flat_m), 100)
    else:
        bins = np.linspace(np.min(flat_m), np.max(flat_m), int(par.bins))
    counts = kde.evaluate(bins)

    gap = 3
    max_val = counts.max()
    max_ind = counts.argmax()

    min_id0 = np.max([max_ind - gap, 0])
    min_id1 = np.min([max_ind + gap, counts.size - 1])
    while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
        min_id0 -= 1
    while min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]:
        min_id1 += 1

    fit_param = [min_id0, min_id1, max_ind, flat_m.size]
    fit_data = [bins, counts]
    flag_min, r_2_min, popt_min, perr_min = perform_gauss_fit(
        fit_param, fit_data, "Min"
    )

    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val / 2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val / 2:
        half_id1 += 1

    fit_param = [half_id0, half_id1, max_ind, flat_m.size]
    fit_data = [bins, counts]
    flag_half, r_2_half, popt_half, perr_half = perform_gauss_fit(
        fit_param, fit_data, "Half"
    )

    r_2 = r_2_min
    if flag_min == 1 and flag_half == 0:
        popt = popt_min
        perr = perr_min
    elif flag_min == 0 and flag_half == 1:
        popt = popt_half
        perr = perr_half
        r_2 = r_2_half
    elif flag_min * flag_half == 1:
        if r_2_min >= r_2_half:
            popt = popt_min
            perr = perr_min
        else:
            popt = popt_half
            perr = perr_half
            r_2 = r_2_half
    else:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print("\tWARNING: this fit is not converging.", file=dump)
        return None

    state = StateUni(popt[0], popt[1], popt[2])
    state.build_boundaries(par.number_of_sigmas)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"\tmu = {state.mean:.4f} ({perr[0]:.4f}), "
            f"sigma = {state.sigma:.4f} ({perr[1]:.4f}), "
            f"area = {state.area:.4f} ({perr[2]:.4f})",
            file=dump,
        )
        print(f"\tFit r2 = {r_2}", file=dump)

    return state


def find_stable_trj(
    cl_ob: ClusteringObject1D,
    state: StateUni,
    tmp_labels: np.ndarray,
    lim: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Identification of windows contained in a certain state.

    Parameters
    ----------

    cl_ob : ClusteringObject1D
        The clustering object.

    state : StateUni
        The state we want to put windows in.

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

    remaning_data = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window * tau_window : (window + 1) * tau_window]
        remaning_data.append(r_w)
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


def fit_local_maxima(
    cl_ob: ClusteringObject1D,
    m_clean: np.ndarray,
    par: Parameters,
    tmp_labels: np.ndarray,
    lim: int,
):
    """
    This functions takes care of particular cases where the
    data points on the tails of a Gaussian are not correctly assigned,
    creating weird sharp peaks in the hsistogram.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    m_clean : np.ndarray
        Input data for Gaussian fitting.

    par : Parameters
        Object containing parameters for fitting.

    tmp_labels : np.ndarray
        Labels indicating window classifications.

    tau_windw : int
        The time resolution of the analysis.

    lim : int
        Offset value for classifying stable windows.
    """
    flat_m = m_clean.flatten()

    kde = gaussian_kde(flat_m)
    if par.bins == "auto":
        bins = np.linspace(np.min(flat_m), np.max(flat_m), 100)
    else:
        bins = np.linspace(np.min(flat_m), np.max(flat_m), int(par.bins))
    counts = kde.evaluate(bins)

    gap = 3

    max_ind, _ = scipy.signal.find_peaks(counts)
    max_val = np.array([counts[i] for i in max_ind])

    for i, m_ind in enumerate(max_ind[:1]):
        min_id0 = np.max([m_ind - gap, 0])
        min_id1 = np.min([m_ind + gap, counts.size - 1])
        while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
            min_id0 -= 1
        while (
            min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]
        ):
            min_id1 += 1

        fit_param = [min_id0, min_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_min, r_2_min, popt_min, perr_min = perform_gauss_fit(
            fit_param, fit_data, "Min"
        )

        half_id0 = np.max([m_ind - gap, 0])
        half_id1 = np.min([m_ind + gap, counts.size - 1])
        while half_id0 > 0 and counts[half_id0] > max_val[i] / 2:
            half_id0 -= 1
        while half_id1 < counts.size - 1 and counts[half_id1] > max_val[i] / 2:
            half_id1 += 1

        fit_param = [half_id0, half_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_half, r_2_half, popt_half, perr_half = perform_gauss_fit(
            fit_param, fit_data, "Half"
        )

        r_2 = r_2_min
        if flag_min == 1 and flag_half == 0:
            popt = popt_min
            perr = perr_min
        elif flag_min == 0 and flag_half == 1:
            popt = popt_half
            perr = perr_half
            r_2 = r_2_half
        elif flag_min * flag_half == 1:
            if r_2_min >= r_2_half:
                popt = popt_min
                perr = perr_min
            else:
                popt = popt_half
                perr = perr_half
                r_2 = r_2_half
        else:
            continue

        state = StateUni(popt[0], popt[1], popt[2])
        state.build_boundaries(par.number_of_sigmas)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tmu = {state.mean:.4f} ({perr[0]:.4f}), "
                f"sigma = {state.sigma:.4f} ({perr[1]:.4f}), "
                f"area = {state.area:.4f} ({perr[2]:.4f})",
                file=dump,
            )
            print(f"\tFit r2 = {r_2}", file=dump)

        number_of_windows = tmp_labels.shape[1]

        mask_unclassified = tmp_labels < 0.5
        m_reshaped = cl_ob.data.matrix[
            :, : number_of_windows * par.tau_w
        ].reshape(cl_ob.data.matrix.shape[0], number_of_windows, par.tau_w)
        mask_inf = np.min(m_reshaped, axis=2) >= state.th_inf[0]
        mask_sup = np.max(m_reshaped, axis=2) <= state.th_sup[0]
        mask = mask_unclassified & mask_inf & mask_sup

        tmp_labels[mask] = lim + 1
        counter = np.sum(mask)

        remaning_data = []
        mask_remaining = mask_unclassified & ~mask
        for i, window in np.argwhere(mask_remaining):
            r_w = cl_ob.data.matrix[
                i, window * par.tau_w : (window + 1) * par.tau_w
            ]
            remaning_data.append(r_w)

        if tmp_labels.size == 0:
            return None, None, None, None

        window_fraction = counter / tmp_labels.size

        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(
                f"\tFraction of windows in state {lim + 1}"
                f" = {window_fraction:.3}",
                file=dump,
            )

        m2_array = np.array(remaning_data)
        one_last_state = True
        if len(m2_array) == 0:
            one_last_state = False

        return state, m2_array, window_fraction, one_last_state

    return None, None, None, None


def iterative_search(
    cl_ob: ClusteringObject1D,
) -> Tuple[ClusteringObject1D, np.ndarray, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Parameters
    ----------

    cl_ob : ClusteringObject1D)
        The clustering object.

    Returns
    -------

    cl_ob : ClusteringObject1D
        Updated with the clustering results.

    atl : ndarray
        Temporary array of labels.

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

    if m_copy.shape[1] < cl_ob.par.tau_w:
        cl_ob.state_list = []
        return cl_ob, tmp_labels, env_0

    while True:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
            print(f"- Iteration {iteration_id - 1}", file=dump)
        state = gauss_fit_max(m_copy, cl_ob.par)

        if state is None:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                print(
                    "- Iterations interrupted because "
                    "fit does not converge.",
                    file=dump,
                )
            break

        m_next, counter, env_0 = find_stable_trj(
            cl_ob, state, tmp_labels, states_counter
        )

        if counter <= 0.0:
            state, m_next, counter, env_0 = fit_local_maxima(
                cl_ob,
                m_copy,
                cl_ob.par,
                tmp_labels,
                states_counter,
            )

            if counter == 0 or state is None:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
                    print(
                        "- Iterations interrupted because last state is empty.",
                        file=dump,
                    )
                break

        state.perc = counter
        states_list.append(state)
        states_counter += 1
        iteration_id += 1
        m_copy = m_next

    cl_ob.iterations = len(states_list)

    atl, lis = relabel_states(tmp_labels, states_list)
    cl_ob.state_list = lis

    return cl_ob, atl, env_0


def timeseries_analysis(
    cl_ob: ClusteringObject1D, tau_w: int
) -> Tuple[int, float, List[float]]:
    """
    The clustering analysis to compute the dependence on time resolution.

    Parameters
    ----------

    cl_ob : ClusteringObject1D
        The clustering object.

    tau_w : int
        The time resolution for the analysis.

    Returns
    -------

    n_states : int
        Number of identified states.

    fraction_0 : float
        Fraction of unclassified data points.

    Notes
    -----

    - Creates a copy of the clustering object and of the parameters
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
        return 0, 1.0, [1.0]

    list_of_states, tmp_labels = set_final_states(
        tmp_cl_ob.state_list,
        tmp_labels,
        AREA_MAX_OVERLAP,
    )

    tmp_cl_ob.data.labels, tmp_cl_ob.state_list = max_prob_assignment(
        list_of_states,
        tmp_cl_ob.data.matrix,
        tmp_labels,
        tmp_cl_ob.data.range,
        tau_w,
        tmp_cl_ob.par.number_of_sigmas,
    )

    list_of_pop = [state.perc for state in tmp_cl_ob.state_list]
    fraction_0 = 1 - np.sum(list_of_pop)
    list_of_pop.insert(0, fraction_0)
    n_states = len(tmp_cl_ob.state_list)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print(
            f"* Number of states identified: {n_states} [{fraction_0}]\n",
            file=dump,
        )

    del tmp_cl_ob
    return n_states, fraction_0, list_of_pop


def full_output_analysis(cl_ob: ClusteringObject1D):
    """
    The complete clustering analysis with the input parameters.

    Parameters
    ----------

    cl_ob : ClusteringObject1D
        The clustering object.

    Notes
    -----

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

    list_of_states, tmp_labels = set_final_states(
        cl_ob.state_list,
        tmp_labels,
        AREA_MAX_OVERLAP,
    )

    cl_ob.data.labels, cl_ob.state_list = max_prob_assignment(
        list_of_states,
        cl_ob.data.matrix,
        tmp_labels,
        cl_ob.data.range,
        cl_ob.par.tau_w,
        cl_ob.par.number_of_sigmas,
    )

    cl_ob.data.labels = cl_ob.create_all_the_labels()


def time_resolution_analysis(cl_ob: ClusteringObject1D):
    """
    Explore parameter space and compute the dependence on time resolution.

    Parameters
    ----------

    cl_ob : ClusteringObject1D
        The clustering object.

    Notes
    -----

    - Generates the parameters' grid
    - Performs and stores the clustering for all the parameters' combinations
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
    list_of_pop: List[List[float]] = [[] for _ in tau_window_list]

    for i, tau_w in enumerate(tau_window_list):
        n_s, f_0, l_pop = timeseries_analysis(cl_ob, tau_w)
        number_of_states.append(n_s)
        fraction_0.append(f_0)
        list_of_pop[i] = l_pop

    cl_ob.number_of_states = np.array(number_of_states)
    cl_ob.fraction_0 = np.array(fraction_0)
    cl_ob.list_of_pop = list_of_pop


def main(
    matrix: np.ndarray,
    tau_window: int,
    tau_window_list: List[int],
    bins: Union[int, str],
    number_of_sigmas: float,
) -> ClusteringObject1D:
    """
    Returns the clustering object with the analysis.

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

    Returns
    -------
    clustering_object : ClusteringObject1D
        The final clustering object.

    Notes
    -----
    - Reads the data and the parameters
    - Performs the quick analysis for all the values in tau_window_list
    - Performs a detailed analysis with the selected tau_window
    """
    clustering_object = all_the_input_stuff(
        matrix,
        tau_window,
        tau_window_list,
        bins,
        number_of_sigmas,
    )

    time_resolution_analysis(clustering_object)

    full_output_analysis(clustering_object)

    return clustering_object
