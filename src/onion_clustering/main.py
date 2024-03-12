"""
Code for clustering of univariate time-series data.
See the documentation for all the details.
"""

import copy
import os
import shutil
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from onion_clustering.classes import ClusteringObject1D
from onion_clustering.first_classes import Parameters, StateUni, UniData
from onion_clustering.functions import (
    gaussian,
    moving_average,
    param_grid,
    plot_histo,
    read_input_data,
    relabel_states,
    set_final_states,
)

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = "states_output.txt"


def all_the_input_stuff() -> ClusteringObject1D:
    """
    Data preprocessing for the analysis.

    Returns:
    - ClusteringObject1D

    - Reads analysis parameters
    - Reads input raw data
    - Removes initial 't_delay' frames
    - Creates blank files and directories for output
    - Creates and returns the ClusteringObject1D for the analysis
    """
    par = Parameters("input_parameters.txt")
    par.print_to_screen()

    data_directory = read_input_data()
    if isinstance(data_directory, str):
        data = UniData(data_directory)
    else:
        print("\tERROR: data_directory.txt is missing or wrongly formatted. ")

    data.remove_delay(par.t_delay)

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

    clustering_object = ClusteringObject1D(par, data)

    return clustering_object


def perform_gauss_fit(
    param: List[int], data: List[np.ndarray], int_type: str
) -> Tuple[bool, int, np.ndarray]:
    """
    Gaussian fit on the data histogram.

    Args:
    - param (List[int]): a list of the parameters for the fit:
        initial index,
        final index,
        index of the max,
        amount of data points,
        gap value for histogram smoothing
    - data (List[np.ndarray]): a list of the data for the fit
        histogram binning,
        histogram counts
    - int_type (str): the type of the fitting interval ('max' or 'half')

    Returns:
    - A boolean value for the fit convergence
    - goodness (int): the fit quality (max is 5)
    - popt (np.ndarray): the optimal gaussians fit parameters

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
        print("\t" + int_type + " fit: Runtime error. ")
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )
    except TypeError:
        print("\t" + int_type + " fit: TypeError.")
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )
    except ValueError:
        print("\t" + int_type + " fit: ValueError.")
        return (
            False,
            goodness,
            np.empty(
                3,
            ),
        )


def gauss_fit_max(
    m_clean: np.ndarray, par: Parameters, filename: str, full_out: bool
) -> Union[StateUni, None]:
    """
    Selection of the optimal interval and parameters in order to fit a state.

    Args:
    - m_clean (np.ndarray): the data points
    - par (Parameters): object containing parameters for the analysis.
    - filename (str): name of the output plot file
    - full_out (bool): activates the full output printing

    Returns:
    - state (StateUni): object containing Gaussian fit parameters
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
        print(
            "\tWARNING: there are empty bins. "
            "Consider reducing the number of bins."
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
        print("\tWARNING: this fit is not converging.")
        return None

    state = StateUni(popt[0], popt[1], popt[2])
    state.build_boundaries(NUMBER_OF_SIGMAS)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        print("\n", file=file)
        print(
            f"\tmu = {state.mean:.4f}, sigma = {state.sigma:.4f},"
            f" area = {state.area:.4f}"
        )
        print(
            f"\tmu = {state.mean:.4f}, sigma = {state.sigma:.4f},"
            f" area = {state.area:.4f}",
            file=file,
        )
        print("\tFit goodness = " + str(goodness), file=file)

    if full_out:
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
    Identification of windows contained in a certain state.

    Args:
    - cl_ob (ClusteringObject1D): the clustering object
    - state (StateUni): the state
    - tmp_labels (np.ndarray): contains the cluster labels of all the
        signal windows
    - lim (int): the algorithm iteration

    Returns:
    - m2_array (np.ndarray): array of still unclassified data points
    - window_fraction (float): fraction of windows classified in this state
    - env_0 (bool): indicates if there are still unclassified data points

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
    cl_ob: ClusteringObject1D, name: str, full_out: bool
) -> Tuple[ClusteringObject1D, np.ndarray, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Args:
    - cl_ob (ClusteringObject1D): the clustering object
    - name (str): name for output figures
    - full_out (bool): activates the full output printing

    Returns:
    - cl_ob (ClusteringObject1D): updated with the clustering results
    - atl (np.ndarray): temporary array of labels
    - env_0 (bool): indicates if there are unclassified data points

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
        state = gauss_fit_max(
            m_copy,
            cl_ob.par,
            "output_figures/" + name + "Fig1_" + str(iteration_id),
            full_out,
        )

        if state is None:
            print("Iterations interrupted because fit does not converge. ")
            break

        m_next, counter, env_0 = find_stable_trj(
            cl_ob, state, tmp_labels, states_counter
        )

        state.perc = counter
        states_list.append(state)
        states_counter += 1
        iteration_id += 1
        if counter <= 0.0:
            print("Iterations interrupted because last state is empty. ")
            break

        m_copy = m_next

    cl_ob.iterations = len(states_list)

    atl, lis = relabel_states(tmp_labels, states_list)
    cl_ob.states = lis

    return cl_ob, atl, env_0


def timeseries_analysis(
    cl_ob: ClusteringObject1D, tau_w: int, t_smooth: int, full_out: bool
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
    - Otherwise, final states are identified by "set_final_states"
    - Number of states and fraction of unclassified points are computed
    """
    print("* New analysis: ", tau_w, t_smooth)
    name = str(t_smooth) + "_" + str(tau_w) + "_"

    tmp_cl_ob = copy.deepcopy(cl_ob)
    tmp_cl_ob.par.tau_w = tau_w
    tmp_cl_ob.par.t_smooth = t_smooth

    tmp_cl_ob.preparing_the_data()
    if full_out:
        tmp_cl_ob.plot_input_data(name + "Fig0")

    tmp_cl_ob, tmp_labels, one_last_state = iterative_search(
        tmp_cl_ob, name, full_out
    )

    if len(tmp_cl_ob.states) == 0:
        print("* No possible classification was found. ")
        del tmp_cl_ob
        return 1, 1.0

    tmp_cl_ob.states, tmp_cl_ob.data.labels = set_final_states(
        tmp_cl_ob.states, tmp_labels, tmp_cl_ob.data.range
    )

    fraction_0 = 1 - np.sum([state.perc for state in tmp_cl_ob.states])
    n_states = len(tmp_cl_ob.states)
    if one_last_state:
        n_states += 1
    print(f"Number of states identified: {n_states}, [{fraction_0}]")

    del tmp_cl_ob
    return n_states, fraction_0


def full_output_analysis(
    cl_ob: ClusteringObject1D, full_out: bool
) -> ClusteringObject1D:
    """
    The complete clustering analysis with the input parameters.

    Args:
    - cl_ob (ClusteringObject1D): the clustering object
    - full_out (bool): activates the full output printing

    Returns:
    - cl_ob (ClusteringObject1D): the upodated clustering object,
        with the clustering resutls

    - Preprocesses the data
    - Performs the clustering with the iterative search and classification
    - If no classification is found, return
    - Otherwise, final states are identified by "set_final_states"
    """
    cl_ob.preparing_the_data()

    cl_ob, tmp_labels, _ = iterative_search(cl_ob, "", full_out)

    if len(cl_ob.states) == 0:
        print("* No possible classification was found. ")
        return cl_ob

    cl_ob.states, cl_ob.data.labels = set_final_states(
        cl_ob.states, tmp_labels, cl_ob.data.range
    )

    return cl_ob


def time_resolution_analysis(cl_ob: ClusteringObject1D, full_out: bool):
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
    tau_window_list, t_smooth_list = param_grid(
        cl_ob.par, cl_ob.data.num_of_steps
    )

    number_of_states = []
    fraction_0 = []
    for tau_w in tau_window_list:
        tmp = [tau_w]
        tmp1 = [tau_w]
        for t_s in t_smooth_list:
            n_s, f_0 = timeseries_analysis(cl_ob, tau_w, t_s, full_out)
            tmp.append(n_s)
            tmp1.append(f_0)
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


def main(full_output: bool = True) -> ClusteringObject1D:
    """
    Returns the clustering object with the analysis.

    Args:
    - full_output (bool): activates the full output printing

    Returns:
    - clustering_object (ClusteringObject1D): the final clustering object

    - Reads the data and the parameters
    - Explore the parameters (tau_window, t_smooth) space
    - Performs a detailed analysis with the selected parameters
    """
    print("##############################################################")
    print("# If you publish results using onion-clustering, please cite #")
    print("# this work: https://doi.org/10.48550/arXiv.2402.07786.      #")
    print("##############################################################")

    clustering_object = all_the_input_stuff()

    time_resolution_analysis(clustering_object, full_output)

    clustering_object = full_output_analysis(clustering_object, full_output)

    return clustering_object


if __name__ == "__main__":
    main()
