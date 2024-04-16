"""
Should contains all the functions in common between the 2 codes.
"""

from typing import List, Tuple

import numpy as np
import scipy.optimize
import scipy.signal
from onion_clustering._internal.first_classes import StateMulti, StateUni

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

    Notes
    -----

    If the array is 1D, applies the moving average. If the array is 2D,
    applies the moving average on the axis=1.
    """
    weights = np.ones(window) / window
    if data.ndim == 1:
        return np.convolve(data, weights, mode="valid")
    if data.ndim >= 2:
        return np.apply_along_axis(
            lambda x: np.convolve(x, weights, mode="valid"), axis=1, arr=data
        )
    raise ValueError(
        "Invalid array dimension. Only 1D and 2D arrays are supported."
    )


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
    gap: int,
    m_limits: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
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

    gap : int
        Minimum allowed gap size for fitting intervals.

    m_limits : List[List[int]]
        List of min and max of the data along each dimension.

    Returns
    -------

    flag : int
        Flag indicating the success (1) or failure (0) of the fitting process.

    goodness : int
        Goodness value representing the fitting quality (5 is the maximum).

    popt : ndarray
        Optimal values for the parameters (mu, sigma, area) of the fitted
        Gaussian.

    Notes
    -----

    - Initializes flag and goodness variables
    - Selects the data on which the fit will be performed
    - Compute initial guess for the params
    - Tries the fit. If it converges:
        - computes the fit quality by checking some requirements
        - else, returns (0, 5, np.empty(3))
    """
    flag = 1
    goodness = 5

    edges_selection = edges[minima[2 * dim] : minima[2 * dim + 1]]
    all_axes = tuple(i for i in range(counts.ndim) if i != dim)
    counts_selection = np.sum(counts, axis=all_axes)
    counts_selection = counts_selection[minima[2 * dim] : minima[2 * dim + 1]]

    mu0 = edges[max_ind]
    sigma0 = (edges[minima[2 * dim + 1]] - edges[minima[2 * dim]]) / 2
    area0 = max(counts_selection) * np.sqrt(np.pi) * sigma0

    try:
        popt, pcov, _, _, _ = scipy.optimize.curve_fit(
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

        # Check goodness of fit and update the goodness variable
        if popt[0] < edges_selection[0] or popt[0] > edges_selection[-1]:
            goodness -= 1
        if popt[1] > edges_selection[-1] - edges_selection[0]:
            goodness -= 1
        if popt[2] < area0 / 2:
            goodness -= 1

        # Calculate parameter errors
        perr = np.sqrt(np.diag(pcov))
        for j, par_err in enumerate(perr):
            if par_err / popt[j] > 0.5:
                goodness -= 1

        # Check if the fitting interval is too small in either dimension
        if minima[2 * dim + 1] - minima[2 * dim] <= gap:
            goodness -= 1
    except RuntimeError:
        print("\tFit: Runtime error. ")
        flag = 0
        popt = np.empty((3,))
    except TypeError:
        print("\tFit: TypeError.")
        flag = 0
        popt = np.empty((3,))
    except ValueError:
        print("\tFit: ValueError.")
        flag = 0
        popt = np.empty((3,))

    return flag, goodness, popt


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


def set_final_states(
    list_of_states: List[StateUni],
    all_the_labels: np.ndarray,
    m_range: np.ndarray,
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
            if j != i:
                if (
                    st_0.peak > st_1.peak
                    and abs(st_1.mean - st_0.mean) < st_0.sigma
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
            list_of_distances = [
                np.linalg.norm(
                    list_of_states[pair[1]].mean - list_of_states[pair[0]].mean
                )
                for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmin(list_of_distances)])

    # Settle merging chains
    # if [i, j], all the [k, i] become [k, j]
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0]:
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

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(
        all_the_labels.shape
    )

    final_map = np.zeros(max(np.unique(all_the_labels)) + 1, dtype=int)
    for i, el in enumerate(np.unique(all_the_labels)):
        final_map[el] = i
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

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    updated_states[0].th_inf[0] = m_range[0]
    updated_states[0].th_inf[1] = 0

    for i in range(len(updated_states) - 1):
        th_val, th_type = find_intersection(
            updated_states[i], updated_states[i + 1]
        )
        updated_states[i].th_sup[0] = th_val
        updated_states[i].th_sup[1] = th_type
        updated_states[i + 1].th_inf[0] = th_val
        updated_states[i + 1].th_inf[1] = th_type

    updated_states[-1].th_sup[0] = m_range[1]
    updated_states[-1].th_sup[1] = 0

    if updated_states[0].th_sup[0] < m_range[0]:
        updated_states.pop(0)
        updated_states[0].th_inf[0] = m_range[0]
        mask = all_the_labels > 1
        all_the_labels[mask] -= 1

    if updated_states[-1].th_inf[0] > m_range[1]:
        updated_states.pop(-1)
        updated_states[-1].th_inf[1] = m_range[1]
        mask = all_the_labels == np.max(all_the_labels)
        all_the_labels[mask] -= 1

    with open(OUTPUT_FILE, "a", encoding="utf-8") as dump:
        print("* FINAL STATES AND THRESHOLDS:", file=dump)
        print("# Mu \t Sigma \t A \t state_fraction", file=dump)
        for state in updated_states:
            print(state.mean, state.sigma, state.area, state.perc, file=dump)
        print("# threshold type", file=dump)
        for state in updated_states:
            print(state.th_inf[0], state.th_inf[1], file=dump)
        print(
            updated_states[-1].th_sup[0],
            updated_states[-1].th_sup[1],
            file=dump,
        )

    return updated_states, all_the_labels


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
            list_of_distances = [
                np.linalg.norm(
                    sorted_states[pair[1]].mean - sorted_states[pair[0]].mean
                )
                for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmin(list_of_distances)])

    # Settle merging chains
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0]:
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
            centers = "[" + str(state.mean[0]) + ", "
            for tmp in state.mean[1:-1]:
                centers += str(tmp) + ", "
            centers += str(state.mean[-1]) + "]"
            axis = "[" + str(state.axis[0]) + ", "
            for tmp in state.axis[1:-1]:
                axis += str(tmp) + ", "
            axis += str(state.axis[-1]) + "]"
            print(centers, axis, state.perc, file=dump)

    return all_the_labels, updated_states
