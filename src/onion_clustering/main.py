"""
Code for clustering of univariate time-series data.
See the documentation for all the details.
"""

import copy
import os
import shutil
import warnings
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.optimize import OptimizeWarning

from onion_clustering.classes import ClusteringObject1D
from onion_clustering.first_classes import Parameters, StateUni, UniData
from onion_clustering.functions import (
    gaussian,
    max_prob_assignment,
    moving_average,
    param_grid,
    plot_histo,
    read_input_data,
    relabel_states,
    set_final_states,
)

OUTPUT_FILE = "states_output.txt"
AREA_MAX_OVERLAP = 0.8


def all_the_input_stuff(number_of_sigmas: float) -> ClusteringObject1D:
    """
    Reads input parameters and raw data from specified files and directories,
    processes the raw data, and creates output files.

    Parameters
    ----------

    number_of_sigmas : float
        The signal windos are classified inside a state with a certain mean
        and std_dev if all the points differ from the mean less than
        std_dev * number_of_sigmas.

    Returns
    -------

    clustering_object : ClusteringObject1D
        The object containing all the information and data for the analysis.
    """
    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters("input_parameters.txt")
    par.print_to_screen()

    # Read raw data from the specified directory/files.
    if isinstance(data_directory, str):
        data = UniData(data_directory)
    else:
        print("\tERROR: data_directory.txt is missing or wrongly formatted. ")

    # Remove initial frames based on 't_delay'.
    data.remove_delay(par.t_delay)

    ### Create files for output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        print("#", file=file)
    figures_folder = "output_figures"
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    for filename in os.listdir(figures_folder):
        file_path = os.path.join(figures_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as ex_msg:
            print(f"Failed to delete {file_path}. Reason: {ex_msg}")

    clustering_object = ClusteringObject1D(par, data, number_of_sigmas)

    return clustering_object


def perform_gauss_fit(
    param: List[int],
    data: List[np.ndarray],
    int_type: str,
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Perform Gaussian fit on given data.

    Parameters
    ----------

    id0 : int
        Index representing the lower limit for data selection.

    id1 : int
        Index representing the upper limit for data selection.

    bins : np.ndarray
        Array containing bin values.

    counts : np.ndarray
        Array containing counts corresponding to bins.

    n_data : int
        Number of data points.

    interval_type : str
        Type of interval ("min" of "half").

    Returns
    -------

    bool
        True if the fit is successful, False otherwise.

    coeff_det_r2 : float
        Coefficient of determination of the fit. Zero if the fit fails.

    popt : np.ndarray of shape (3,)
        Parameters of the Gaussian fit if successful, zeros otherwise.

    perr : np.ndarray of shape (3,)
        Uncertanties on the Gaussian fit if successful, zeros otherwise.
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
    sigma0 = (bins[id1] - bins[id0]) / 6
    area0 = counts[max_ind] * np.sqrt(np.pi) * sigma0
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
                popt[1] *= -1
                popt[2] *= -1
            popt[2] *= n_data
            perr = np.array([np.sqrt(pcov[i][i]) for i in range(popt.size)])
            perr[2] *= n_data
            ss_res = np.sum(infodict["fvec"] ** 2)
            ss_tot = np.sum((selected_counts - np.mean(selected_counts)) ** 2)
            coeff_det_r2 = 1 - ss_res / ss_tot
            flag = True
    except OptimizeWarning:
        print(f"\t{int_type} fit: Optimize warning. ")
    except RuntimeError:
        print(f"\t{int_type} fit: Runtime error. ")
    except TypeError:
        print(f"\t{int_type} fit: TypeError.")
    except ValueError:
        print(f"\t{int_type} fit: ValueError.")

    return flag, coeff_det_r2, popt, perr


def gauss_fit_max(
    m_clean: np.ndarray,
    par: Parameters,
    number_of_sigmas: float,
    filename: str,
    full_out: bool,
) -> Union[StateUni, None]:
    """
    Performs Gaussian fitting on input data.

    Parameters
    ----------

    m_clean : np.ndarray
        Input data for Gaussian fitting.

    par : Parameters
        Object containing parameters for fitting.

    number_of_sigmas : float
        To set the thresholds for assigning windows to the state.

    filename : str
        Name of the output plot file.

    full_output : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    Returns
    -------

    state : StateUni
        Object containing Gaussian fit parameters (mu, sigma, area)
        or None if the fit fails.
    """
    print("* Gaussian fit...")
    flat_m = m_clean.flatten()

    ### 1. Histogram ###
    counts, bins = np.histogram(flat_m, bins=par.bins, density=True)
    gap = 1
    if bins.size > 49:
        gap = int(bins.size * 0.02) * 2
    print(f"\tNumber of bins = {bins.size}, gap = {gap}")

    ### 2. Smoothing with tau = 3 ###
    counts = moving_average(counts, gap)
    bins = moving_average(bins, gap)
    if (counts == 0.0).any():
        print(
            "\tWARNING: there are empty bins. "
            "Consider reducing the number of bins."
        )

    ### 3. Find the maximum ###
    max_val = counts.max()
    max_ind = counts.argmax()

    ### 4. Find the minima surrounding it ###
    min_id0 = np.max([max_ind - gap, 0])
    min_id1 = np.min([max_ind + gap, counts.size - 1])
    while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
        min_id0 -= 1
    while min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]:
        min_id1 += 1

    ### 5. Try the fit between the minima and check its goodness ###
    fit_param = [min_id0, min_id1, max_ind, flat_m.size]
    fit_data = [bins, counts]
    flag_min, r_2_min, popt_min, perr_min = perform_gauss_fit(
        fit_param, fit_data, "Min"
    )

    ### 6. Find the inrterval of half height ###
    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val / 2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val / 2:
        half_id1 += 1

    ### 7. Try the fit between the minima and check its goodness ###
    fit_param = [half_id0, half_id1, max_ind, flat_m.size]
    fit_data = [bins, counts]
    flag_half, r_2_half, popt_half, perr_half = perform_gauss_fit(
        fit_param, fit_data, "Half"
    )

    ### 8. Choose the best fit ###
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
        print("\tWARNING: this fit is not converging.")
        return None

    state = StateUni(popt[0], popt[1], popt[2])
    state.build_boundaries(number_of_sigmas)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        print("\n", file=file)
        print(
            f"\tmu = {state.mean:.4f} ({perr[0]:.4f}), "
            f"sigma = {state.sigma:.4f} ({perr[1]:.4f}), "
            f"area = {state.area:.4f} ({perr[2]:.4f})"
        )
        print(f"\tFit r2 = {r_2}")
        print(
            f"\tmu = {state.mean:.4f} ({perr[0]:.4f}), "
            f"sigma = {state.sigma:.4f} ({perr[1]:.4f}), "
            f"area = {state.area:.4f} ({perr[2]:.4f})",
            file=file,
        )
        print(f"\tFit r2 = {r_2}", file=file)

    if full_out:
        ### Plot the distribution and the fitted gaussians
        y_spread = np.max(m_clean) - np.min(m_clean)
        y_lim = [
            np.min(m_clean) - 0.025 * y_spread,
            np.max(m_clean) + 0.025 * y_spread,
        ]
        fig, axes = plt.subplots()
        plot_histo(axes, counts, bins)
        axes.set_xlim(y_lim)
        tmp_popt = [state.mean, state.sigma, state.area / flat_m.size]
        axes.plot(
            np.linspace(bins[0], bins[-1], 1000),
            gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt),
        )

        fig.savefig(filename + ".png", dpi=600)
        plt.close(fig)

    return state


def find_stable_trj(
    cl_ob: ClusteringObject1D,
    state: StateUni,
    tmp_labels: np.ndarray,
    lim: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Identifies stable windows within a state.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    state : StateUni
        Object containing stable state parameters.

    all_the_labels : np.ndarray
        Labels indicating window classifications.

    offset : int
        Offset value for classifying stable windows.

    Returns
    -------

    m2_array : np.ndarray
        Array of non-stable windows.

    fw : float
        Fraction of windows classified as stable.

    one_last_state : bool
        Indicates if there's one last state remaining.
    """
    print("* Finding stable windows...")

    # Calculate the number of windows in the trajectory
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

    # Initialize an empty list to store non-stable windows
    remaning_data = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window * tau_window : (window + 1) * tau_window]
        remaning_data.append(r_w)

    # Calculate the fraction of stable windows found
    window_fraction = counter / (tmp_labels.size)

    # Print the fraction of stable windows
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        print(
            f"\tFraction of windows in state {lim + 1}"
            f" = {window_fraction:.3}"
        )
        print(
            f"\tFraction of windows in state {lim + 1}"
            f" = {window_fraction:.3}",
            file=file,
        )

    # Convert the list of non-stable windows to a NumPy array
    m2_array = np.array(remaning_data)
    one_last_state = True
    if m2_array.size == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows,
    # and the updated list_of_states
    return m2_array, window_fraction, one_last_state


def solve_batman(
    cl_ob: ClusteringObject1D,
    m_clean: np.ndarray,
    par: Parameters,
    number_of_sigmas: float,
    tmp_labels: np.ndarray,
    tau_window: int,
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

    number_of_sigmas : float
        To set the thresholds for assigning windows to the state.

    tmp_labels : np.ndarray
        Labels indicating window classifications.

    tau_windw : int
        The time resolution of the analysis.

    lim : int
        Offset value for classifying stable windows.
    """
    flat_m = m_clean.flatten()

    ### 1. Histogram ###
    counts, bins = np.histogram(flat_m, bins=par.bins, density=True)
    gap = 1
    if bins.size > 99:
        gap = int(bins.size * 0.02)
    print(f"\tNumber of bins = {bins.size}, gap = {gap}")

    ### 2. Smoothing with tau = 3 ###
    counts = moving_average(counts, gap)
    bins = moving_average(bins, gap)
    if (counts == 0.0).any():
        print(
            "\tWARNING: there are empty bins. "
            "Consider reducing the number of bins."
        )

    ### 3. Find the maxima ###
    max_ind, _ = scipy.signal.find_peaks(counts)
    max_val = np.array([counts[i] for i in max_ind])

    for i, m_ind in enumerate(max_ind[:1]):
        ### 4. Find the minima surrounding it ###
        min_id0 = np.max([m_ind - gap, 0])
        min_id1 = np.min([m_ind + gap, counts.size - 1])
        while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
            min_id0 -= 1
        while (
            min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]
        ):
            min_id1 += 1

        ### 5. Try the fit between the minima and check its goodness ###
        fit_param = [min_id0, min_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_min, r_2_min, popt_min, _ = perform_gauss_fit(
            fit_param, fit_data, "Min"
        )

        ### 6. Find the inrterval of half height ###
        half_id0 = np.max([m_ind - gap, 0])
        half_id1 = np.min([m_ind + gap, counts.size - 1])
        while half_id0 > 0 and counts[half_id0] > max_val[i] / 2:
            half_id0 -= 1
        while half_id1 < counts.size - 1 and counts[half_id1] > max_val[i] / 2:
            half_id1 += 1

        ### 7. Try the fit between the minima and check its goodness ###
        fit_param = [half_id0, half_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_half, r_2_half, popt_half, _ = perform_gauss_fit(
            fit_param, fit_data, "Half"
        )

        ### 8. Choose the best fit ###
        if flag_min == 1 and flag_half == 0:
            popt = popt_min
        elif flag_min == 0 and flag_half == 1:
            popt = popt_half
        elif flag_min * flag_half == 1:
            if r_2_min >= r_2_half:
                popt = popt_min
            else:
                popt = popt_half
        else:
            continue

        state = StateUni(popt[0], popt[1], popt[2])
        state.build_boundaries(number_of_sigmas)

        # Calculate the number of windows in the trajectory
        number_of_windows = tmp_labels.shape[1]

        mask_unclassified = tmp_labels < 0.5
        m_reshaped = cl_ob.data.matrix[
            :, : number_of_windows * tau_window
        ].reshape(cl_ob.data.matrix.shape[0], number_of_windows, tau_window)
        mask_inf = np.min(m_reshaped, axis=2) >= state.th_inf[0]
        mask_sup = np.max(m_reshaped, axis=2) <= state.th_sup[0]
        mask = mask_unclassified & mask_inf & mask_sup

        tmp_labels[mask] = lim + 1
        counter = np.sum(mask)

        # Initialize an empty list to store non-stable windows
        remaning_data = []
        mask_remaining = mask_unclassified & ~mask
        for i, window in np.argwhere(mask_remaining):
            r_w = cl_ob.data.matrix[
                i, window * tau_window : (window + 1) * tau_window
            ]
            remaning_data.append(r_w)

        # Calculate the fraction of stable windows found
        window_fraction = counter / (tmp_labels.size)

        # Print the fraction of stable windows
        with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
            print(
                f"\tFraction of windows in state {lim + 1}"
                f" = {window_fraction:.3}"
            )
            print(
                f"\tFraction of windows in state {lim + 1}"
                f" = {window_fraction:.3}",
                file=file,
            )

        # Convert the list of non-stable windows to a NumPy array
        m2_array = np.array(remaning_data)
        one_last_state = True
        if len(m2_array) == 0:
            one_last_state = False

        # Return the array of non-stable windows, the fraction of stable windows,
        # and the updated list_of_states
        return state, m2_array, window_fraction, one_last_state

    # If event trying all the maxima does not work, surrend
    return None, None, None, None


def iterative_search(
    cl_ob: ClusteringObject1D,
    name: str,
    full_out: bool,
) -> Tuple[ClusteringObject1D, np.ndarray, bool]:
    """
    Performs an iterative search for stable states in a trajectory.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    name : str
        Name for identifying output figures.

    full_output : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    Returns
    -------

    cl_ob : ClusteringObject1D
        Updated with the clustering results.

    atl : np.ndarray of shape (n_particles, n_windows)
        Temporary array of labels.

    one_last_state : bool
        Indicates if there's one last state remaining.
    """

    # Initialize an array to store labels for each window.
    num_windows = int(cl_ob.data.num_of_steps / cl_ob.par.tau_w)
    tmp_labels = np.zeros((cl_ob.data.num_of_particles, num_windows)).astype(
        int
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        print(f"tau_window = {cl_ob.par.tau_w}", file=file)

    states_list = []
    m_copy = cl_ob.data.matrix
    iteration_id = 1
    states_counter = 0
    one_last_state = False
    while True:
        ### Locate and fit maximum in the signal distribution
        state = gauss_fit_max(
            m_copy,
            cl_ob.par,
            cl_ob.number_of_sigmas,
            f"output_figures/{name}Fig1_{iteration_id}",
            full_out,
        )
        if state is None:
            print("Iterations interrupted because fit does not converge. ")
            break

        ### Find the windows in which the trajectories are stable
        m_next, counter, one_last_state = find_stable_trj(
            cl_ob, state, tmp_labels, states_counter
        )

        ### Exit the loop if no new stable windows are found
        if counter <= 0.0:
            state, m_next, counter, one_last_state = solve_batman(
                cl_ob,
                m_copy,
                cl_ob.par,
                cl_ob.number_of_sigmas,
                tmp_labels,
                cl_ob.par.tau_w,
                states_counter,
            )

            if counter == 0 or state is None:
                print("Iterations interrupted because last state is empty. ")
                break

        state.perc = counter
        states_list.append(state)
        states_counter += 1
        iteration_id += 1
        m_copy = m_next

        if m_next.size == 0:
            print("Iterations interrupted because all points are classififed.")
            break

    cl_ob.iterations = len(states_list)
    atl, lis = relabel_states(tmp_labels, states_list)
    cl_ob.states = lis
    return cl_ob, atl, one_last_state


def timeseries_analysis(
    cl_ob: ClusteringObject1D,
    tau_w: int,
    t_smooth: int,
    full_out: bool,
) -> Tuple[int, float, List[float]]:
    """
    Performs an analysis pipeline on time series data.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    tau_w : int
        The time resolution for the analysis.

    t_smooth : int
        The width of the moving average for the analysis.

    Returns
    -------

    num_states : int
        Number of identified states.

    fraction_0 : float
        Fraction of unclassified data points. Between 0 and 1.

    list_of_pop : List[float]
        List of the populations of the different states.
    """
    print("* New analysis: ", tau_w, t_smooth)
    name = str(t_smooth) + "_" + str(tau_w) + "_"

    tmp_cl_ob = copy.deepcopy(cl_ob)
    tmp_cl_ob.par.tau_w = tau_w
    tmp_cl_ob.par.t_smooth = t_smooth

    tmp_cl_ob.preparing_the_data()
    tmp_cl_ob.plot_input_data(name + "Fig0")

    tmp_cl_ob, tmp_labels, _ = iterative_search(tmp_cl_ob, name, full_out)

    if len(tmp_cl_ob.states) == 0:
        print("* No possible classification was found. ")
        # We need to free the memory otherwise it accumulates
        del tmp_cl_ob
        return 0, 1.0, [1.0]

    list_of_states, tmp_labels = set_final_states(
        tmp_cl_ob.states,
        tmp_labels,
        AREA_MAX_OVERLAP,
    )

    tmp_cl_ob.data.labels, tmp_cl_ob.states = max_prob_assignment(
        list_of_states,
        tmp_cl_ob.data.matrix,
        tmp_labels,
        tmp_cl_ob.data.range,
        tau_w,
        tmp_cl_ob.number_of_sigmas,
    )

    list_of_pop = [state.perc for state in tmp_cl_ob.states]
    fraction_0 = 1 - np.sum(list_of_pop)
    list_of_pop.insert(0, fraction_0)
    n_states = len(tmp_cl_ob.states)

    # We need to free the memory otherwise it accumulates
    del tmp_cl_ob

    print(f"Number of states identified: {n_states}, [{fraction_0}]\n")
    return n_states, fraction_0, list_of_pop


def full_output_analysis(
    cl_ob: ClusteringObject1D,
    full_out: bool,
) -> ClusteringObject1D:
    """
    Perform a comprehensive analysis on the input data.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    full_out : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    Returns
    -------

    cl_ob : ClusteringObject1D
        Updated with the clustering results.
    """

    cl_ob.preparing_the_data()

    cl_ob, tmp_labels, _ = iterative_search(cl_ob, "", full_out)
    if len(cl_ob.states) == 0:
        print("* No possible classification was found. ")
        return cl_ob

    list_of_states, tmp_labels = set_final_states(
        cl_ob.states,
        tmp_labels,
        AREA_MAX_OVERLAP,
    )

    cl_ob.data.labels, cl_ob.states = max_prob_assignment(
        list_of_states,
        cl_ob.data.matrix,
        tmp_labels,
        cl_ob.data.range,
        cl_ob.par.tau_w,
        cl_ob.number_of_sigmas,
    )

    return cl_ob


def time_resolution_analysis(
    cl_ob: ClusteringObject1D,
    full_out: bool,
):
    """
    Performs Temporal Resolution Analysis (TRA) to explore parameter
    space and analyze the dataset.

    Parameters
    ----------
    cl_ob : ClusteringObject1D

    full_out : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.
    """
    tau_window_list, t_smooth_list = param_grid(
        cl_ob.par, cl_ob.data.num_of_steps
    )
    cl_ob.tau_window_list = np.array(tau_window_list)
    cl_ob.t_smooth_list = np.array(t_smooth_list)

    number_of_states = []
    fraction_0 = []
    list_of_pop: List[List[List[float]]] = [
        [[] for _ in tau_window_list] for _ in t_smooth_list
    ]

    for i, tau_w in enumerate(tau_window_list):
        tmp = [tau_w]
        tmp1 = [tau_w]
        for j, t_s in enumerate(t_smooth_list):
            n_s, f_0, l_pop = timeseries_analysis(cl_ob, tau_w, t_s, full_out)
            tmp.append(n_s)
            tmp1.append(f_0)
            list_of_pop[j][i] = l_pop
        number_of_states.append(tmp)
        fraction_0.append(tmp1)

    number_of_states_arr = np.array(number_of_states)
    fraction_0_arr = np.array(fraction_0)

    np.savetxt(
        "number_of_states.txt",
        number_of_states,
        fmt="%i",
        delimiter="\t",
        header="tau_window\t number_of_states for different t_smooth",
    )
    np.savetxt(
        "fraction_0.txt",
        fraction_0,
        delimiter=" ",
        header="tau_window\t fraction in ENV0 for different t_smooth",
    )

    cl_ob.number_of_states = number_of_states_arr
    cl_ob.fraction_0 = fraction_0_arr
    cl_ob.list_of_pop = list_of_pop


def main(
    full_output: bool = True,
    number_of_sigmas: float = 2.0,
) -> ClusteringObject1D:
    """
    Returns the clustering object with the analysi.

    Parameters
    ----------

    full_output : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    number_of_sigmas : float
        The signal windos are classified inside a state with a certain mean
        and std_dev if all the points differ from the mean less than
        std_dev * number_of_sigmas.

    Returns
    -------

    clustering_object : ClusteringObject1D
        The clusteriong object, with the input data, the parameters and the
        results of the clustering.

    Notes
    -----

    all_the_input_stuff() reads the data and the parameters

    time_resolution_analysis() explore the parameter
        (tau_window, t_smooth) space.

    full_output_analysis() performs a detailed analysis
        with the chosen parameters.
    """
    print("##############################################################")
    print("# If you publish results using onion-clustering, please cite #")
    print("# this work: https://doi.org/10.48550/arXiv.2402.07786.      #")
    print("##############################################################")

    clustering_object = all_the_input_stuff(number_of_sigmas)
    time_resolution_analysis(clustering_object, full_output)
    clustering_object = full_output_analysis(clustering_object, full_output)

    return clustering_object


if __name__ == "__main__":
    main()
