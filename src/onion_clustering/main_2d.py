"""
Code for clustering of multivariate (2- or 3-dimensional) time-series data.
See the documentation for all the details.
"""

import copy
import os
import shutil
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage
from matplotlib.patches import Ellipse

from onion_clustering.classes import ClusteringObject2D
from onion_clustering.first_classes import MultiData, Parameters, StateMulti
from onion_clustering.functions import (
    custom_fit,
    find_half_height_around_max,
    find_minima_around_max,
    moving_average_2d,
    param_grid,
    read_input_data,
    relabel_states_2d,
)

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = "states_output.txt"


def all_the_input_stuff() -> ClusteringObject2D:
    """
    Data preprocessing for the analysis.

    Returns:
    - ClusteringObject2D

    - Reads analysis parameters
    - Reads input raw data
    - Removes initial 't_delay' frames
    - Creates blank files and directories for output
    - Creates and returns the ClusteringObject2D for the analysis
    """
    par = Parameters("input_parameters.txt")
    par.print_to_screen()

    data_directory = read_input_data()
    data = MultiData(data_directory)
    data.remove_delay(par.t_delay)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("#")
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
        except OSError as exc_msg:
            print(f"Failed to delete {file_path}. Reason: {exc_msg}")

    clustering_object = ClusteringObject2D(par, data)

    return clustering_object


def gauss_fit_max(
    m_clean: np.ndarray,
    m_limits: np.ndarray,
    bins: Union[int, str],
    filename: str,
    full_out: bool,
) -> Union[StateMulti, None]:
    """
    Selection of the optimal region and parameters in order to fit a state.

    Args:
    - m_clean (np.ndarray): the data points
    - m_limits (np.ndarray): the min and max of the data points
    - bins (Union[int, str]): the histogram binning
    - filename (str): name of the output plot file
    - full_out (bool): activates the full output printing

    Returns:
    - state (StateMulti): object containing Gaussian fit parameters
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
        print("\tWARNING: this fit is not converging.")
        return None
    if len(popt) != m_clean.shape[2] * 3:
        print("\tWARNING: this fit is not converging.")
        return None

    mean, sigma, area = [], [], []
    for dim in range(m_clean.shape[2]):
        mean.append(popt[3 * dim])
        sigma.append(popt[3 * dim + 1])
        area.append(popt[3 * dim + 2])
    state = StateMulti(np.array(mean), np.array(sigma), np.array(area))
    state.build_boundaries(NUMBER_OF_SIGMAS)

    if m_clean.shape[2] == 2:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
            print("\n", file=file)
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}],"
                f" sigma = [{popt[1]:.4f}, {popt[4]:.4f}]"
            )
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}],"
                f" sigma = [{popt[1]:.4f}, {popt[4]:.4f}],"
                f" area = {popt[2]:.4f}, {popt[5]:.4f}",
                file=file,
            )
            print("\tFit goodness = " + str(goodness), file=file)

        if full_out:
            fig, ax = plt.subplots(figsize=(6, 6))
            img = NonUniformImage(ax, interpolation="nearest")
            xcenters = (edges[0][:-1] + edges[0][1:]) / 2
            ycenters = (edges[1][:-1] + edges[1][1:]) / 2
            img.set_data(xcenters, ycenters, counts.T)
            ax.add_image(img)
            ax.scatter(mean[0], mean[1], s=8.0, c="red")
            circle1 = Ellipse(
                tuple(mean), sigma[0], sigma[1], color="r", fill=False
            )
            circle2 = Ellipse(
                tuple(mean),
                state.axis[0],
                state.axis[1],
                color="r",
                fill=False,
            )
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.set_xlim(m_limits[0][0], m_limits[0][1])
            ax.set_ylim(m_limits[1][0], m_limits[1][1])

            fig.savefig(filename + ".png", dpi=600)
            plt.close(fig)
    elif m_clean.shape[2] == 3:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
            print("\n", file=file)
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], "
                f"sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], "
                f"area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}"
            )
            print(
                f"\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], "
                f"sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], "
                f"area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}",
                file=file,
            )
            print("\tFit goodness = " + str(goodness), file=file)

        if full_out:
            fig, ax = plt.subplots(2, 2, figsize=(6, 6))
            xcenters = (edges[0][:-1] + edges[0][1:]) / 2
            ycenters = (edges[1][:-1] + edges[1][1:]) / 2
            zcenters = (edges[2][:-1] + edges[2][1:]) / 2

            img = NonUniformImage(ax[0][0], interpolation="nearest")
            img.set_data(xcenters, ycenters, np.sum(counts, axis=0))
            ax[0][0].add_image(img)
            ax[0][0].scatter(mean[0], mean[1], s=8.0, c="red")
            circle1 = Ellipse(
                tuple([mean[0], mean[1]]),
                sigma[0],
                sigma[1],
                color="r",
                fill=False,
            )
            circle2 = Ellipse(
                tuple([mean[0], mean[1]]),
                state.axis[0],
                state.axis[1],
                color="r",
                fill=False,
            )
            ax[0][0].add_patch(circle1)
            ax[0][0].add_patch(circle2)

            img = NonUniformImage(ax[0][1], interpolation="nearest")
            img.set_data(zcenters, ycenters, np.sum(counts, axis=1))
            ax[0][1].add_image(img)
            ax[0][1].scatter(mean[2], mean[1], s=8.0, c="red")
            circle1 = Ellipse(
                tuple([mean[2], mean[1]]),
                sigma[2],
                sigma[1],
                color="r",
                fill=False,
            )
            circle2 = Ellipse(
                tuple([mean[2], mean[1]]),
                state.axis[2],
                state.axis[1],
                color="r",
                fill=False,
            )
            ax[0][1].add_patch(circle1)
            ax[0][1].add_patch(circle2)

            img = NonUniformImage(ax[1][0], interpolation="nearest")
            img.set_data(xcenters, zcenters, np.sum(counts, axis=2))
            ax[1][0].add_image(img)
            ax[1][0].scatter(mean[0], mean[2], s=8.0, c="red")
            circle1 = Ellipse(
                tuple([mean[0], mean[2]]),
                sigma[0],
                sigma[2],
                color="r",
                fill=False,
            )
            circle2 = Ellipse(
                tuple([mean[0], mean[2]]),
                state.axis[0],
                state.axis[2],
                color="r",
                fill=False,
            )
            ax[1][0].add_patch(circle1)
            ax[1][0].add_patch(circle2)

            ax[0][0].set_xlim(m_limits[0][0], m_limits[0][1])
            ax[0][0].set_ylim(m_limits[1][0], m_limits[1][1])
            ax[0][1].set_xlim(m_limits[2][0], m_limits[2][1])
            ax[0][1].set_ylim(m_limits[1][0], m_limits[1][1])
            ax[1][0].set_xlim(m_limits[0][0], m_limits[0][1])
            ax[1][0].set_ylim(m_limits[2][0], m_limits[2][1])

            fig.savefig(filename + ".png", dpi=600)
            plt.close(fig)

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
    - cl_ob (ClusteringObject2D): the clustering object
    - state (StateMulti): the state
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
    m2_arr = np.array(remaning_data)

    env_0 = True
    if len(m2_arr) == 0:
        env_0 = False

    return m2_arr, window_fraction, env_0


def iterative_search(
    cl_ob: ClusteringObject2D, name: str, full_out: bool
) -> Tuple[ClusteringObject2D, bool]:
    """
    Iterative search for stable windows in the trajectory.

    Args:
    - cl_ob (ClusteringObject2D): the clustering object
    - name (str): name for output figures
    - full_out (bool): activates the full output printing

    Returns:
    - cl_ob (ClusteringObject1D): updated with the clustering results
    - env_0 (bool): indicates if there are unclassified data points

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
        print(f"* Iteration {iteration_id - 1}")
        state = gauss_fit_max(
            m_copy,
            np.array(cl_ob.data.range),
            bins,
            "output_figures/" + name + "Fig1_" + str(iteration_id),
            full_out,
        )

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
            print("* Iterations interrupted because last state is empty. ")
            break
        if m_new.size == 0:
            print(
                "* Iterations interrupted because all data "
                "points assigned. "
            )
            break
        m_copy = m_new

    cl_ob.iterations = len(states_list)

    all_the_labels, list_of_states = relabel_states_2d(tmp_labels, states_list)
    cl_ob.data.labels = all_the_labels
    cl_ob.states = list_of_states

    return cl_ob, env_0


def timeseries_analysis(
    cl_ob: ClusteringObject2D, tau_w: int, t_smooth: int, full_out: bool
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

    print("* New analysis: ", tau_w, t_smooth)
    name = str(t_smooth) + "_" + str(tau_w) + "_"

    tmp_cl_ob = copy.deepcopy(cl_ob)
    tmp_cl_ob.par.tau_w = tau_w
    tmp_cl_ob.par.t_smooth = t_smooth

    tmp_cl_ob.preparing_the_data()
    if full_out:
        tmp_cl_ob.plot_input_data(name + "Fig0")

    tmp_cl_ob, one_last_state = iterative_search(tmp_cl_ob, name, full_out)

    if len(tmp_cl_ob.states) == 0:
        print("* No possible classification was found. ")
        del tmp_cl_ob
        return 1, 1.0

    fraction_0 = 1 - np.sum([state.perc for state in tmp_cl_ob.states])
    n_states = len(tmp_cl_ob.states)
    if one_last_state:
        n_states += 1
    print(f"Number of states identified: {n_states}, [{fraction_0}]")

    del tmp_cl_ob
    return n_states, fraction_0


def full_output_analysis(
    cl_ob: ClusteringObject2D, full_out: bool
) -> ClusteringObject2D:
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
    cl_ob.preparing_the_data()

    cl_ob, _ = iterative_search(cl_ob, "", full_out)

    if len(cl_ob.states) == 0:
        print("* No possible classification was found. ")
        return cl_ob

    return cl_ob


def time_resolution_analysis(cl_ob: ClusteringObject2D, full_out: bool):
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


def main(full_output: bool = True) -> ClusteringObject2D:
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

    clustering_object = all_the_input_stuff()

    time_resolution_analysis(clustering_object, full_output)

    clustering_object = full_output_analysis(clustering_object, full_output)

    return clustering_object


if __name__ == "__main__":
    main()
