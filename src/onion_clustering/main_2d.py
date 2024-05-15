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

OUTPUT_FILE = "states_output.txt"


def all_the_input_stuff(number_of_sigmas: float) -> ClusteringObject2D:
    """
    Reads input parameters and raw data from specified files and directories,
    processes the raw data, and creates output files.
    """
    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters("input_parameters.txt")
    par.print_to_screen()

    data = MultiData(data_directory)
    data.remove_delay(par.t_delay)

    ### Create files for output
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

    clustering_object = ClusteringObject2D(par, data, number_of_sigmas)

    return clustering_object


def gauss_fit_max(
    m_clean: np.ndarray,
    m_limits: np.ndarray,
    bins: Union[int, str],
    number_of_sigmas: float,
    filename: str,
    full_out: bool,
) -> Union[StateMulti, None]:
    """
    Perform Gaussian fit and generate plots based on the provided data.

    Parameters
    ----------

    m_clean : np.ndarray of shape (n_particles, n_timesteps)
        Processed array of cleaned data.

    m_limits : list[list[int]]
        List containing minimum and maximum values for each dimension.

    bins : Union[int, str]
        Number of bins for histograms or 'auto' for automatic binning.

    number_of_sigmas : float
        To set the thresholds for assigning windows to the state.

    filename : str
        Name of the output file to save the plot.

    full_out : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    Returns
    -------

    State : StateMulti
        Object containing Gaussian fit parameters (mu, sigma, area)
        or None if the fit fails.
    """
    print("* Gaussian fit...")
    flat_m = m_clean.reshape(
        (m_clean.shape[0] * m_clean.shape[1], m_clean.shape[2])
    )

    ### 1. Histogram with 'auto' binning ###
    if bins == "auto":
        bins = max(int(np.power(m_clean.size, 1 / 3) * 2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)
    gap = 1
    edges_sides = np.array([e.size for e in edges])
    if np.all(edges_sides > 49):
        # gap = 3
        gap = int(np.min(edges_sides) * 0.02) * 2
        if gap % 2 == 0:
            gap += 1

    ### 2. Smoothing with gap ###
    counts = moving_average_2d(counts, gap)

    ### 3. Find the maximum ###
    def find_max_index(data: np.ndarray):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    max_ind = find_max_index(counts)

    ### 4. Find the minima surrounding it ###
    minima = find_minima_around_max(counts, max_ind, gap)

    ### 5. Try the fit between the minima and check its quality ###
    popt_min: List[float] = []
    det_coeff_min = 0.0
    for dim in range(m_clean.shape[2]):
        try:
            flag_min, r2_min, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_min.extend(popt)
            det_coeff_min += r2_min
        except RuntimeError:
            popt_min = []
            flag_min = False

    ### 6. Find the interval of half height ###
    minima = find_half_height_around_max(counts, max_ind, gap)

    ### 7. Try the fit between the minima and check its quality ###
    popt_half: List[float] = []
    det_coeff_half = 0.0
    for dim in range(m_clean.shape[2]):
        try:
            flag_half, r2_half, popt = custom_fit(
                dim, max_ind[dim], minima, edges[dim], counts, m_limits
            )
            popt[2] *= flat_m.T[0].size
            popt_half.extend(popt)
            det_coeff_half += r2_half
        except RuntimeError:
            popt_half = []
            flag_half = False

    ### 8. Choose the best fit ###
    r2 = det_coeff_min
    if flag_min == 1 and flag_half == 0:
        popt = np.array(popt_min)
    elif flag_min == 0 and flag_half == 1:
        popt = np.array(popt_half)
        r2 = det_coeff_half
    elif flag_min * flag_half == 1:
        if det_coeff_min >= det_coeff_half:
            popt = np.array(popt_min)
        else:
            popt = np.array(popt_half)
            r2 = det_coeff_half
    else:
        print("\tWARNING: this fit is not converging.")
        return None

    if len(popt) != m_clean.shape[2] * 3:
        print("\tWARNING: this fit is not converging.")
        return None

    ### Find the tresholds for state identification
    mean, sigma, area = [], [], []
    for dim in range(m_clean.shape[2]):
        mean.append(popt[3 * dim])
        sigma.append(popt[3 * dim + 1])
        area.append(popt[3 * dim + 2])
    state = StateMulti(np.array(mean), np.array(sigma), np.array(area))
    state.build_boundaries(number_of_sigmas)

    ### Plot the distribution and the fitted Gaussians
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
            print(f"\tFit r2 = {r2}", file=file)

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
            print(f"\tFit r2 = {r2}", file=file)

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
    m_clean: np.ndarray,
    tau_window: int,
    state: StateMulti,
    all_the_labels: np.ndarray,
    offset: int,
) -> Tuple[np.ndarray, float, bool]:
    """
    Find stable windows in the trajectory.

    Args:
    - m_clean (np.ndarray): Cleaned trajectory data.
    - tau_window (int): Length of the window for analysis.
    - state (StateMulti): State information.
    - all_the_labels (np.ndarray): All labels for the trajectory.
    - offset (int): Offset for labeling stable windows.

    Returns a Tuple of:
    - np.ndarray: Array of non-stable windows.
    - float: Fraction of stable windows.
    - bool: Indicates one last state after finding stable windows.
    """
    print("* Finding stable windows...")

    # Calculate the number of windows in the trajectory
    number_of_windows = int(m_clean.shape[1] / tau_window)

    mask_unclassified = all_the_labels < 0.5
    m_reshaped = m_clean[:, : number_of_windows * tau_window].reshape(
        m_clean.shape[0], number_of_windows, tau_window, m_clean.shape[2]
    )
    shifted = m_reshaped - state.mean
    rescaled = shifted / state.axis
    squared_distances = np.sum(rescaled**2, axis=3)
    mask_dist = np.max(squared_distances, axis=2) <= 1.0
    mask = mask_unclassified & mask_dist

    all_the_labels[mask] = offset + 1
    counter = np.sum(mask)

    # Store non-stable windows in a list, for the next iteration
    m_new = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window * tau_window : (window + 1) * tau_window]
        m_new.append(r_w)

    # Calculate the fraction of stable windows found
    fraction_of_points = counter / (all_the_labels.size)

    # Print the fraction of stable windows
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        print(
            f"\tFraction of windows in state {offset + 1}"
            f" = {fraction_of_points:.3}"
        )
        print(
            f"\tFraction of windows in state {offset + 1}"
            f" = {fraction_of_points:.3}",
            file=file,
        )

    # Convert the list of non-stable windows to a NumPy array
    m_new_arr = np.array(m_new)
    one_last_state = True
    if len(m_new_arr) == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows,
    # and the updated list_of_states
    return m_new_arr, fraction_of_points, one_last_state


def iterative_search(
    cl_ob: ClusteringObject2D,
    name: str,
    full_out: bool,
) -> Tuple[ClusteringObject2D, bool]:
    """
    Perform an iterative search to identify stable windows in trajectory data.

    Args:
    - name (str): Name for the output figures.

    Returns:
    - cl_ob (ClusteringObject): updated with the clustering results.
    - one_last_state (bool): Indicates if there's one last state remaining.
    """
    tau_w, bins = cl_ob.par.tau_w, cl_ob.par.bins

    # Initialize an array to store labels for each window.
    num_windows = int(cl_ob.data.num_of_steps / tau_w)
    tmp_labels = np.zeros((cl_ob.data.num_of_particles, num_windows)).astype(
        int
    )

    states_list = []
    m_copy = cl_ob.data.matrix
    iteration_id = 1
    states_counter = 0
    one_last_state = False
    while True:
        ### Locate and fit maximum in the signal distribution
        state = gauss_fit_max(
            m_copy,
            np.array(cl_ob.data.range),
            bins,
            cl_ob.number_of_sigmas,
            "output_figures/" + name + "Fig1_" + str(iteration_id),
            full_out,
        )
        if state is None:
            print("Iterations interrupted because fit does not converge. ")
            break

        ### Find the windows in which the trajectories are stable
        m_new, counter, one_last_state = find_stable_trj(
            cl_ob.data.matrix, tau_w, state, tmp_labels, states_counter
        )
        state.perc = counter

        if counter > 0.0:
            states_list.append(state)

        states_counter += 1
        iteration_id += 1
        ### Exit the loop if no new stable windows are found
        if counter <= 0.0:
            print("Iterations interrupted because last state is empty. ")
            break
        if m_new.size == 0:
            print("Iterations interrupted because all data points assigned. ")
            break
        m_copy = m_new

    cl_ob.iterations = len(states_list)
    all_the_labels, list_of_states = relabel_states_2d(tmp_labels, states_list)
    cl_ob.data.labels = all_the_labels
    cl_ob.states = list_of_states
    return cl_ob, one_last_state


def timeseries_analysis(
    cl_ob: ClusteringObject2D,
    tau_w: int,
    t_smooth: int,
    full_out: bool,
) -> Tuple[int, float, List[float]]:
    """
    Perform time series analysis on the input data.

    Parameters
    ----------

    cl_ob : ClusteringObject2D

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

    tmp_cl_ob, _ = iterative_search(tmp_cl_ob, name, full_out)

    if len(tmp_cl_ob.states) == 0:
        print("* No possible classification was found. ")
        # We need to free the memory otherwise it accumulates
        del tmp_cl_ob
        return 0, 1.0, [1.0]

    list_of_pop = [state.perc for state in tmp_cl_ob.states]
    fraction_0 = 1 - np.sum(list_of_pop)
    list_of_pop.insert(0, fraction_0)
    n_states = len(tmp_cl_ob.states)

    # We need to free the memory otherwise it accumulates
    del tmp_cl_ob

    print(f"Number of states identified: {n_states}, [{fraction_0}]\n")
    return n_states, fraction_0, list_of_pop


def full_output_analysis(
    cl_ob: ClusteringObject2D,
    full_out: bool,
) -> ClusteringObject2D:
    """
    Perform a comprehensive analysis on the input data.

    Parameters
    ----------

    cl_ob : ClusteringObject2D

    full_out : bool
        If True, plot all the intermediate histograms with the best fit.
        Useful for debugging.

    Returns
    -------

    cl_ob : ClusteringObject1D
        Updated with the clustering results.
    """
    cl_ob.preparing_the_data()

    cl_ob, _ = iterative_search(cl_ob, "", full_out)
    if len(cl_ob.states) == 0:
        print("* No possible classification was found. ")
        return cl_ob

    return cl_ob


def time_resolution_analysis(cl_ob: ClusteringObject2D, full_out: bool):
    """
    Performs Temporal Resolution Analysis (TRA) to explore parameter
    space and analyze the dataset.

    Args:
    - cl_ob (ClusteringObject): Conteining now only the raw input data.
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
) -> ClusteringObject2D:
    """
    Returns the clustering object with the analysi.

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
