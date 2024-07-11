"""
Should contains all the functions in common between the 2 codes.
"""

from typing import List, Tuple

import numpy as np
import scipy.optimize
import scipy.signal
from onion_clustering._internal.first_classes import StateMulti, StateUni
from scipy.integrate import quad
from scipy.optimize import OptimizeWarning

OUTPUT_FILE = "onion_clustering_log.txt"


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average on a 1D or 2D np.ndarray.

    Parameters
    ----------

    data : np.ndarray
        The input array to be smoothed.

    window : int
        The size of the moving average window.

    Returns
    -------

    np.ndarray
        The smoothed array.
    """
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode="valid")


def moving_average_2d(data: np.ndarray, side: int) -> np.ndarray:
    """
    2D moving average on an np.ndarray.

    Parameters
    ----------

    data : np.ndarray of shape (a, b)
        The 2D input array to be smoothed.

    side : int
        The side length of the square moving average window (must be
        an odd number).

    Returns
    -------

    np.ndarray
        The smoothed array.

    Notes
    -----

    Checks if the side is an odd number. Averages over the square centered
    in each element.
    """
    if side % 2 == 0:
        raise ValueError("L must be an odd number.")

    half_width = (side - 1) // 2
    result = np.zeros_like(data, dtype=float)
    for index in np.ndindex(*data.shape):
        slices = tuple(
            slice(
                max(0, i - half_width),
                min(data.shape[dim], i + half_width + 1),
            )
            for dim, i in enumerate(index)
        )
        subarray = data[slices]
        if subarray.size > 0:
            result[index] = subarray.mean()

    return result


def param_grid(trj_len: int) -> List[int]:
    """
    Generates tau_window_list.

    Parameters
    ----------

    trj_len : int
        Number of frames in the data.

    Returns
    -------

    tau_window_list : List[int]
        A list of tau_window values.
    """
    tmp = np.geomspace(2, trj_len, num=20, dtype=int)
    tau_window_list = []
    for tau_w in tmp:
        if tau_w not in tau_window_list:
            tau_window_list.append(tau_w)

    return tau_window_list


def gaussian(
    x_points: np.ndarray, x_mean: float, sigma: float, area: float
) -> np.ndarray:
    """Compute the Gaussian function values at given points 'x_points'.

    Parameters
    ----------

    x_points : np.ndarray
        Array of input values.

    x_mean : float
        Mean value of the Gaussian.

    sigma : float
        Rescaled standard deviation of the Gaussian.

    area : float
        Area under the Gaussian.

    Returns
    -------

    np.ndarray
        Gaussian function values computed at the input points.
    """
    return (
        np.exp(-(((x_points - x_mean) / sigma) ** 2))
        * area
        / (np.sqrt(np.pi) * sigma)
    )


def find_minima_around_max(
    data: np.ndarray, max_ind: Tuple[int, ...], gap: int
) -> List[int]:
    """
    Minima surrounding the maximum value in the given data array.

    Parameters
    ----------

    data : np.ndarray
        Input data array.

    max_ind : tuple
        Indices of the maximum value in the data.

    gap : int
        Gap value to determine the search range around the maximum.

    Returns
    -------

    minima : List[int]
        List of indices representing the minima surrounding the maximum
        in each dimension.

    Notes
    -----

    The precise details of this function have to be described.
    """
    minima: List[int] = []

    for dim in range(data.ndim):
        min_id0 = max(max_ind[dim] - gap, 0)
        min_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

        tmp_max1: List[int] = list(max_ind)
        tmp_max2: List[int] = list(max_ind)

        tmp_max1[dim] = min_id0
        tmp_max2[dim] = min_id0 - 1
        while min_id0 > 0 and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]:
            tmp_max1[dim] -= 1
            tmp_max2[dim] -= 1
            min_id0 -= 1

        tmp_max1 = list(max_ind)
        tmp_max2 = list(max_ind)

        tmp_max1[dim] = min_id1
        tmp_max2[dim] = min_id1 + 1
        while (
            min_id1 < data.shape[dim] - 1
            and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]
        ):
            tmp_max1[dim] += 1
            tmp_max2[dim] += 1
            min_id1 += 1

        minima.extend([min_id0, min_id1])

    return minima


def find_half_height_around_max(
    data: np.ndarray, max_ind: Tuple[int, ...], gap: int
) -> List[int]:
    """
    Half-heigth points surrounding the maximum value in the given data array.

    Parameters
    ----------

    data : np.ndarray
        Input data array.

    max_ind : tuple
        Indices of the maximum value in the data.

    gap : int
        Gap value to determine the search range around the maximum.

    Returns
    -------

    minima : List[int]
        List of indices representing the minima surrounding the maximum
        in each dimension.

    Notes
    -----

    The precise details of this function have to be described.
    """
    max_val = data.max()
    minima: List[int] = []

    for dim in range(data.ndim):
        half_id0 = max(max_ind[dim] - gap, 0)
        half_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

        tmp_max: List[int] = list(max_ind)

        tmp_max[dim] = half_id0
        while half_id0 > 0 and data[tuple(tmp_max)] > max_val / 2:
            tmp_max[dim] -= 1
            half_id0 -= 1

        tmp_max = list(max_ind)

        tmp_max[dim] = half_id1
        while (
            half_id1 < data.shape[dim] - 1
            and data[tuple(tmp_max)] > max_val / 2
        ):
            tmp_max[dim] += 1
            half_id1 += 1

        minima.extend([half_id0, half_id1])

    return minima


def custom_fit(
    dim: int,
    max_ind: int,
    minima: list[int],
    edges: np.ndarray,
    counts: np.ndarray,
    m_limits: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """
    Fit a Gaussian curve to selected multivarite data.

    Parameters
    ----------

    dim : int
        The data dimensionality.

    max_ind : int
        Index of the maximum value in the histogram.

    minima : List[int]
        List of indices representing the minimum points.

    edges : ndarray
        Array containing the bin edges of the histogram.

    counts : ndarray
        Array containing histogram counts.

    m_limits : List[List[int]]
        List of min and max of the data along each dimension.

    Returns
    -------

    flag : int
        Flag indicating the success (1) or failure (0) of the fitting process.

    coeff_det_r2 : float
        Determination coefficient of the fit (r^2). Between 0 and 1.

    popt : list[float]
        Optimal values for the parameters (mu, sigma, area) of the
        fitted Gaussian.
    """
    edges_selection = edges[minima[2 * dim] : minima[2 * dim + 1]]
    all_axes = tuple(i for i in range(counts.ndim) if i != dim)
    counts_selection = np.sum(counts, axis=all_axes)
    counts_selection = counts_selection[minima[2 * dim] : minima[2 * dim + 1]]

    mu0 = edges[max_ind]
    sigma0 = (edges[minima[2 * dim + 1]] - edges[minima[2 * dim]]) / 2
    area0 = max(counts_selection) * np.sqrt(np.pi) * sigma0

    flag = 1
    coeff_det_r2 = 0.0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        try:
            popt, _, infodict, _, _ = scipy.optimize.curve_fit(
                gaussian,
                edges_selection,
                counts_selection,
                p0=[mu0, sigma0, area0],
                bounds=(
                    [m_limits[dim][0], 0.0, 0.0],
                    [m_limits[dim][1], np.inf, np.inf],
                ),
                full_output=True,
            )

            ss_res = np.sum(infodict["fvec"] ** 2)
            ss_tot = np.sum(
                (counts_selection - np.mean(counts_selection)) ** 2
            )
            if ss_tot > 0.0:
                coeff_det_r2 = 1.0 - ss_res / ss_tot
            else:
                coeff_det_r2 = 0.0
        except OptimizeWarning:
            print("\tFit: Optimize warning.", file=dump)
            flag = 0
            popt = np.empty((3,))
        except RuntimeError:
            print("\tFit: Runtime error.", file=dump)
            flag = 0
            popt = np.empty((3,))
        except TypeError:
            print("\tFit: TypeError.", file=dump)
            flag = 0
            popt = np.empty((3,))
        except ValueError:
            print("\tFit: ValueError.", file=dump)
            flag = 0
            popt = np.empty((3,))
    return flag, coeff_det_r2, popt


def relabel_states(
    all_the_labels: np.ndarray, states_list: list[StateUni]
) -> Tuple[np.ndarray, list[StateUni]]:
    """
    Relabel states and update the state list.

    Parameters
    ----------

    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels assigned to each window in the trajectory.

    states_list : List[StateUni]
        List of StateUni objects representing different states.

    Returns
    -------

    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels assigned to each window in the trajectory.

    relevant_states : List[StateUni]
        Updated list of StateUni objects representing different states.

    Notes
    -----

    - Removes states with zero relevance
    - Sorts states according to their mean value
    - Creates a dictionary to map old state labels to new ones
    - Relabels the data in all_the_labels according to the new states_list
    """
    relevant_states = [state for state in states_list if state.perc != 0.0]

    relevant_states.sort(key=lambda x: x.mean)

    state_mapping = {
        state_index: index + 1
        for index, state_index in enumerate(
            [states_list.index(state) for state in relevant_states]
        )
    }
    relabel_map = np.zeros(len(states_list) + 1, dtype=int)
    for key, value in state_mapping.items():
        relabel_map[key + 1] = value

    mask = all_the_labels != 0
    all_the_labels[mask] = relabel_map[all_the_labels[mask]]

    return all_the_labels, relevant_states


def find_intersection(st_0: StateUni, st_1: StateUni) -> Tuple[float, int]:
    """
    Finds the intersection between two Gaussians.

    Parameters
    ----------

    st_0, st_1 : StateUni
        The two states we are computing the intersection between.

    Returns
    -------

    th_val : float
        The value of the intersection.

    th_type : int
        The type of the intersection: type 1 if the intersection between the
        two Gaussians exists, type 2 if it does not exist and the weighted
        average between the means is returned.
    """
    coeff_a = st_1.sigma**2 - st_0.sigma**2
    coeff_b = -2 * (st_0.mean * st_1.sigma**2 - st_1.mean * st_0.sigma**2)
    tmp_c = np.log(st_0.area * st_1.sigma / st_1.area / st_0.sigma)
    coeff_c = (
        (st_0.mean * st_1.sigma) ** 2
        - (st_1.mean * st_0.sigma) ** 2
        - ((st_0.sigma * st_1.sigma) ** 2) * tmp_c
    )
    delta = coeff_b**2 - 4 * coeff_a * coeff_c
    if coeff_a == 0.0:
        only_th = (st_0.mean + st_1.mean) / 2 - st_0.sigma**2 / 2 / (
            st_1.mean - st_0.mean
        ) * np.log(st_0.area / st_1.area)
        return only_th, 1
    if delta >= 0:
        th_plus = (-coeff_b + np.sqrt(delta)) / (2 * coeff_a)
        th_minus = (-coeff_b - np.sqrt(delta)) / (2 * coeff_a)
        intercept_plus = gaussian(th_plus, st_0.mean, st_0.sigma, st_0.area)
        intercept_minus = gaussian(th_minus, st_0.mean, st_0.sigma, st_0.area)
        if intercept_plus >= intercept_minus:
            return th_plus, 1
        return th_minus, 1
    th_aver = (st_0.mean / st_0.sigma + st_1.mean / st_1.sigma) / (
        1 / st_0.sigma + 1 / st_1.sigma
    )
    return th_aver, 2


def shared_area_between_gaussians(
    area1, mean1, sigma1, area2, mean2, sigma2
) -> Tuple[float, float]:
    """
    Computes the shared area between two Gaussians.

    Parameters
    ----------

    area1, mean1, sigma1 : float
        The parameters of Gaussian 1.

    area2, mean2, sigma2 : float
        The parameters of Gaussian 2.

    Returns
    -------

    shared_fraction_1 : float
        The fraction of the area of the first Gaussian in common with the
        second Gaussian.

    shared_fraction_2 : float
        The fraction of the area of the second Gaussian in common with the
        first Gaussian.
    """

    def gauss_1(x):
        gauss = gaussian(x, mean1, sigma1, area1)
        return gauss

    def gauss_2(x):
        gauss = gaussian(x, mean2, sigma2, area2)
        return gauss

    def min_of_gaussians(x):
        min_values = np.minimum(
            gaussian(x, mean1, sigma1, area1),
            gaussian(x, mean2, sigma2, area2),
        )
        return min_values

    area_gaussian_1, _ = quad(
        gauss_1, int(mean1 - 3 * sigma1) - 1, int(mean1 + 3 * sigma1) + 1
    )
    area_gaussian_2, _ = quad(
        gauss_2, int(mean2 - 3 * sigma2) - 1, int(mean2 + 3 * sigma2) + 1
    )

    x_min = int(np.min([mean1 - 3 * sigma1, mean2 - 3 * sigma2])) - 1
    x_max = int(np.max([mean1 + 3 * sigma1, mean2 + 3 * sigma2])) + 1
    shared_area, _ = quad(min_of_gaussians, x_min, x_max)

    shared_fraction_1 = shared_area / area_gaussian_1
    shared_fraction_2 = shared_area / area_gaussian_2

    return shared_fraction_1, shared_fraction_2


def final_state_settings(
    list_of_states: List[StateUni],
    m_range: np.ndarray,
) -> List[StateUni]:
    """
    Final adjustemts and output in the list of identified states.

    Parameters
    ----------

    list_of_states : list[StateUni]
        The list of final states.

    m_range : np.ndarray of shape (2,)
        Range of values in the data matrix.

    Returns
    -------

    list_of_states : list[StateUni]
        Now with the correct thresholds asssigned to each state.
    """
    # Calculate the final threshold values
    # and their types based on the intercept between neighboring states.
    if len(list_of_states) == 0:
        return list_of_states

    list_of_states[0].th_inf[0] = m_range[0]
    list_of_states[0].th_inf[1] = 0

    for i in range(len(list_of_states) - 1):
        th_val, th_type = find_intersection(
            list_of_states[i], list_of_states[i + 1]
        )
        list_of_states[i].th_sup[0] = th_val
        list_of_states[i].th_sup[1] = th_type
        list_of_states[i + 1].th_inf[0] = th_val
        list_of_states[i + 1].th_inf[1] = th_type

    list_of_states[-1].th_sup[0] = m_range[1]
    list_of_states[-1].th_sup[1] = 0

    # Write the final states and final thresholds to text files.
    with open("final_states.txt", "a", encoding="utf-8") as file:
        print("####################################", file=file)
        print("# Mu \t Sigma \t A \t state_fraction", file=file)
        for state in list_of_states:
            print(state.mean, state.sigma, state.area, state.perc, file=file)
    with open("final_thresholds.txt", "a", encoding="utf-8") as file:
        print("####################################", file=file)
        print("# Threshold_value \t Threshold type", file=file)
        for state in list_of_states:
            print(state.th_inf[0], state.th_inf[1], file=file)
        print(
            list_of_states[-1].th_sup[0],
            list_of_states[-1].th_sup[1],
            file=file,
        )

    return list_of_states


def set_final_states(
    list_of_states: List[StateUni],
    all_the_labels: np.ndarray,
    area_max_overlap: float,
) -> Tuple[List[StateUni], np.ndarray]:
    """
    Assigns final states and relabels labels based on specific criteria.

    Parameters
    ----------

    list_of_states : List[StateUni]
        List of StateUni objects representing potential states.

    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels for each data point.

    m_range : ndarray of shape (2,)
        Minimum and maximum of values in the data.

    Returns
    -------

    updated_states : List[StateUni]
        The final list of states.

    all_the _labels : ndarray of shape (n_particles, n_windows)
        The final data labels.

    Notes
    -----

    - Merge together the strongly overlapping states.
    - Calculate the final threshold values and their types based on
        the intercept between neighboring states.
    - Write the final states and final thresholds to text files.
    """

    # Find all the possible merges: j could be merged into i --> [j, i]
    proposed_merge = []
    for i, st_0 in enumerate(list_of_states):
        for j, st_1 in enumerate(list_of_states):
            if j > i:
                # Condition 1: area overlap
                shared_area_1, shared_area_2 = shared_area_between_gaussians(
                    st_1.area,
                    st_1.mean,
                    st_1.sigma,
                    st_0.area,
                    st_0.mean,
                    st_0.sigma,
                )
                thresh = area_max_overlap
                if shared_area_1 > thresh >= shared_area_2:
                    proposed_merge.append([j, i])
                elif shared_area_2 > thresh >= shared_area_1:
                    proposed_merge.append([i, j])
                elif shared_area_1 > thresh and shared_area_2 > thresh:
                    proposed_merge.append(
                        [j, i] if shared_area_1 > shared_area_2 else [i, j]
                    )
                # Condition 2: mean proximity
                elif (
                    st_0.peak > st_1.peak
                    and np.abs(st_0.mean - st_1.mean) < st_0.sigma
                    and st_1.sigma < 2 * st_0.sigma
                ):
                    proposed_merge.append([j, i])
                elif (
                    st_1.peak > st_0.peak
                    and np.abs(st_0.mean - st_1.mean) < st_1.sigma
                    and st_0.sigma < 2 * st_1.sigma
                ):
                    proposed_merge.append([i, j])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([pair[0] for pair in proposed_merge])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            importance = [
                list_of_states[pair[1]].perc for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmax(importance)])

    # Settle merging chains
    # if [i, j], all the [k, i] become [k, j]
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0] and elem[0] != pair[1]:
                best_merge[j][1] = pair[1]

    # Relabel the labels in all_the_labels
    relabel_dic = {}
    for pair in best_merge:
        relabel_dic[pair[0]] = pair[1]
    if_env0 = np.any(np.unique(all_the_labels) == 0)

    relabel_map = np.zeros(max(np.unique(all_the_labels) + 1), dtype=int)
    for i, _ in enumerate(relabel_map):
        relabel_map[i] = i
    for key, value in relabel_dic.items():
        relabel_map[key + 1] = value + 1

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(
        all_the_labels.shape
    )

    final_map = np.zeros(max(np.unique(all_the_labels)) + 1, dtype=int)
    for i, el in enumerate(np.unique(all_the_labels)):
        final_map[el] = i + 1 * (1 - if_env0)
    for i, particle in enumerate(all_the_labels):
        for j, el in enumerate(particle):
            all_the_labels[i][j] = final_map[el]

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [
        state
        for i, state in enumerate(list_of_states)
        if i not in states_to_remove
    ]

    return updated_states, all_the_labels


def find_max_prob_state(
    windows: np.ndarray,
    old_labels: np.ndarray,
    list_of_states: List[StateUni],
) -> int:
    """
    Assign multiple windows to the state for which the belonging
    is the most probable in a vectorized manner.

    Parameters
    ----------
    windows : np.ndarray of shape (num_windows, tau_window)
        The signal windows to assign to states.
    old_labels : np.ndarray of shape (num_windows,)
        The temporary labels for the considered signal windows.
    list_of_states : List[StateUni]
        List of the identified states.

    Returns
    -------
    new_labels : np.ndarray of shape (num_windows,)
        The labels for the considered signal windows.
    """
    medians = np.median(windows, axis=1)

    # Prepare arrays for Gaussian calculations
    means = np.array([state.mean for state in list_of_states])
    sigmas = np.array([state.sigma for state in list_of_states])
    areas = np.array([state.area for state in list_of_states])

    # Broadcast medians to shape (num_windows, num_states)
    medians_broadcasted = medians[:, np.newaxis]

    # Calculate Gaussian values for each window's median against all states
    gaussians = (areas / (sigmas * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((medians_broadcasted - means) / sigmas) ** 2
    )

    # Find the state with the maximum Gaussian value for each window
    new_labels = np.argmax(gaussians, axis=1) + 1

    return new_labels


def max_prob_assignment(
    list_of_states: List[StateUni],
    matrix: np.ndarray,
    all_the_labels: np.ndarray,
    m_range: np.ndarray,
    tau_window: int,
    number_of_sigmas: float,
) -> Tuple[np.ndarray, List[StateUni]]:
    """
    After all the states have been identified, assign each window.
    Each signal window is assigned to the most probable state.

    Parameters
    ----------

    list_of_states : List[StateUni]
        List of the identified states.

    matrix : np.ndarray of shape (num_of_particles, num_of_timesteps)
        The data to cluster.

    all_the_labels : np.ndarray of shape (num_of_particles, num_of_windows)
        The temporary labels assigned to the signal windows.

    m_range : np.ndarray of shape (2,)
        Range of values in the data matrix.

    tau_window : int
        The time resolution of the analysis.

    Returns
    -------

    final_labels : np.ndarray of shape (num_of_particles, num_of_windows)
        The definitive labels for all the signal windows.

    updated_states : List[StateUni]
        List of the identified states, with updated percetages.
    """
    N, T = all_the_labels.shape
    reshaped_matrix = matrix[:, : T * tau_window].reshape(N, T, tau_window)
    s_ranges = np.array(
        [number_of_sigmas * state.sigma for state in list_of_states]
    )
    final_labels = np.zeros(all_the_labels.shape, dtype=int)
    mask = all_the_labels > 0

    valid_indices = np.where(mask)
    windows = reshaped_matrix[valid_indices]
    old_labels = all_the_labels[valid_indices]

    new_labels = find_max_prob_state(windows, old_labels, list_of_states)

    # Calculate s_range for the new labels
    s_ranges_selected = s_ranges[new_labels - 1]

    # Check the range condition
    max_values = np.max(windows, axis=1)
    min_values = np.min(windows, axis=1)
    range_conditions = (max_values - min_values) < s_ranges_selected

    # Update final labels based on the range condition
    final_labels[valid_indices] = np.where(range_conditions, new_labels, 0)

    for i, state in enumerate(list_of_states):
        num_of_points = np.sum(final_labels == i + 1)
        state.perc = num_of_points / final_labels.size

    ### To conform to scikit convention, noise has to be labelled "-1"
    final_labels -= 1

    states_to_remove = []
    for i, state in enumerate(list_of_states):
        if state.perc == 0.0:
            states_to_remove.append(i)

    for i in states_to_remove[::-1]:
        list_of_states.pop(i)

    updated_states = final_state_settings(
        list_of_states,
        m_range,
    )

    return final_labels, updated_states


def relabel_states_2d(
    all_the_labels: np.ndarray, states_list: list[StateMulti]
) -> Tuple[np.ndarray, List[StateMulti]]:
    """
    Reorders labels and merges strongly overlapping states in a
    multidimensional space.

    Parameters
    ----------

    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels for each data point.

    states_list : List[StateMulti]
        List of StateUni objects representing potential states.

    Returns
    -------

    all_the_labels : ndarray of shape (n_particles, n_windows)
        The final data labels.

    updated_states: List[StateMulti]
        The final list of states.

    Notes
    -----

    - Remove states with zero relevance.
    - Sort states according to their relevance.
    - Merge together the states which are strongly overlapping.
    - Write the final states to text files.
    """

    sorted_states = [state for state in states_list if state.perc != 0.0]

    # Create a dictionary to map old state labels to new ones
    state_mapping = {
        index: i + 1
        for i, index in enumerate(
            np.argsort([-state.perc for state in sorted_states])
        )
    }

    sorted_states.sort(key=lambda x: x.perc, reverse=True)

    # Relabel the data in all_the_labels according to the new states_list
    mask = all_the_labels != 0  # Create a mask for non-zero elements
    all_the_labels[mask] = np.vectorize(state_mapping.get, otypes=[int])(
        all_the_labels[mask] - 1, 0
    )

    # Find all the possible merges
    proposed_merge = []
    for i, st_0 in enumerate(sorted_states):
        for j, st_1 in enumerate(sorted_states):
            if j > i:
                diff = np.abs(np.subtract(st_1.mean, st_0.mean))
                if np.all(
                    diff
                    < [
                        max(st_0.sigma[k], st_1.sigma[k])
                        for k in range(diff.size)
                    ]
                ):
                    proposed_merge.append([j, i])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([pair[0] for pair in proposed_merge])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            importance = [
                sorted_states[pair[1]].perc for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmax(importance)])

    # Settle merging chains
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0] and elem[0] != pair[1]:
                best_merge[j][1] = pair[1]

    # Relabel the labels in all_the_labels
    relabel_dic = {}
    for pair in best_merge:
        relabel_dic[pair[0]] = pair[1]
    relabel_map = np.zeros(max(np.unique(all_the_labels) + 1), dtype=int)
    for i, _ in enumerate(relabel_map):
        relabel_map[i] = i
    for key, value in relabel_dic.items():
        relabel_map[key + 1] = value + 1

    # Remove the gaps in the labeling
    tmp_labels = np.unique(relabel_map)
    map2 = np.zeros(max(tmp_labels + 1), dtype=int)
    for i, elem in enumerate(tmp_labels):
        map2[elem] = i
    for i, elem in enumerate(relabel_map):
        relabel_map[i] = map2[elem]

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(
        all_the_labels.shape
    )

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [
        state
        for i, state in enumerate(sorted_states)
        if i not in states_to_remove
    ]

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print("- FINAL STATES:", file=dump)
        print("# center_coords, semiaxis, fraction_of_data", file=dump)
        for state in updated_states:
            centers = f"[{state.mean[0]}, "
            for tmp in state.mean[1:-1]:
                centers += f"{tmp}, "
            centers += f"{state.mean[-1]}]"
            axis = f"[{state.axis[0]}, "
            for tmp in state.axis[1:-1]:
                axis += f"{tmp}, "
            axis += f"{state.axis[-1]}]"
            print(centers, axis, state.perc, file=dump)

    return all_the_labels, updated_states
