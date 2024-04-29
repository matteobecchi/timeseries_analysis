"""
Should contains all the functions in common between the 2 codes.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal

from onion_clustering.first_classes import Parameters, StateMulti, StateUni


def read_input_data() -> str:
    """
    Attempt to read the content of 'data_directory.txt' file
    and load it into a NumPy array as strings.
    """
    try:
        data_dir = np.loadtxt("data_directory.txt", dtype=str)
    except OSError as msg_exc:
        print(f"\t Reading data_directory.txt: {msg_exc}")
    except ValueError as msg_exc:
        print(f"\t Reading data_directory.txt: {msg_exc}")

    print("* Reading data from", data_dir)

    return str(data_dir)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Applies a moving average filter to a 1D or 2D NumPy array.

    Args:
    - data (np.ndarray): The input array to be smoothed.
    - window (int): The size of the moving average window.

    Returns:
    - np.ndarray: The smoothed array obtained after applying
    the moving average filter.

    Raises:
    - ValueError: If the input array dimension is not supported
    (only 1D and 2D arrays are supported).
    """

    # Step 1: Create a NumPy array 'weights' with the values 1.0
    # repeated 'window' times.
    # Then, divide each element of 'weights' by the 'window' value
    # to get the average weights.
    weights = np.ones(window) / window

    # Step 2: Apply the moving average filter to the 'data' array using
    # the 'weights' array. The 'np.convolve' function performs a linear
    # convolution between 'data' and 'weights'. The result is a smoothed
    # version of the 'data', where each point represents the weighted
    # average of its neighbors.
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
    """Applies a 2D moving average filter to a NumPy array.

    Args:
    - data (np.ndarray): The 2D input array to be smoothed.
    - side (int): The side length of the square moving average window
        (must be an odd number).

    Returns:
    - np.ndarray: The smoothed array obtained after applying the 2D
        moving average filter.

    Raises:
    - ValueError: If the side length 'side' is not an odd number.
    """

    if side % 2 == 0:  # Check if side is an odd number
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
        # Calculate the average if the subarray is not empty
        if subarray.size > 0:
            result[index] = subarray.mean()

    return result


def plot_histo(ax: plt.Axes, counts: np.ndarray, bins: np.ndarray):
    """Plots a histogram on the specified axes.

    Args:
    - ax: The matplotlib axes to plot on.
    - counts (np.ndarray): The count or frequency of occurrences.
    - bins (np.ndarray): The bin edges defining the intervals.

    Returns:
    - None

    The function plots a histogram with the provided count and bin information
    on the specified axes 'ax' and labels the x and y axes accordingly.
    """

    ax.stairs(counts, bins, fill=True)
    ax.set_xlabel(r"Normalized signal")
    ax.set_ylabel(r"Probability distribution")


def param_grid(par: Parameters, trj_len: int) -> Tuple[List, List]:
    """Generates parameter grids for tau_window and t_smooth.

    Args:
    - par (Parameters): An instance of the Parameters class containing
        parameter details.
    - trj_len (int): Length of the trajectory data.

    Returns:
    - tau_window (List[int]): A list of tau_window values.
    - t_smooth (List[int]): A list of t_smooth values.

    This function generates grids of values for 'tau_window' and 't_smooth'
    based on the provided 'Parameters' instance and the length of the
    trajectory. It calculates the values for 'tau_window' within the range
    defined in 'Parameters' and generates 't_smooth' values within the
    specified range.
    """

    if par.max_tau_w == -1:
        par.max_tau_w = trj_len - par.max_t_smooth
    tmp = np.geomspace(
        par.min_tau_w, par.max_tau_w, num=par.num_tau_w, dtype=int
    )
    tau_window = []
    for tau_w in tmp:
        if tau_w not in tau_window:
            tau_window.append(tau_w)
    print("* Tau_w used:", tau_window)

    t_smooth = list(
        range(par.min_t_smooth, par.max_t_smooth + 1, par.step_t_smooth)
    )
    print("* t_smooth used:", t_smooth)

    return tau_window, t_smooth


def gaussian(
    x_points: np.ndarray, x_mean: float, sigma: float, area: float
) -> np.ndarray:
    """Compute the Gaussian function values at given points 'x'.

    Args:
    - x_points (np.ndarray): Array of input values.
    - x_mean (float): Mean value of the Gaussian function.
    - sigma (float): Standard deviation of the Gaussian function.
    - area (float): Area under the Gaussian curve.

    Returns:
    - np.ndarray: Gaussian function values computed at the input points.

    This function calculates the values of a Gaussian function at the given
    array of points 'x_points' using the provided 'x_mean', 'sigma'
    (standard deviation), and 'area' (area under the curve) parameters.
    It returns an array of Gaussian function values corresponding to the
    input 'x_points'.
    """

    return (
        np.exp(-(((x_points - x_mean) / sigma) ** 2))
        * area
        / (np.sqrt(np.pi) * sigma)
    )


def custom_fit(
    dim: int,
    max_ind: int,
    minima: list[int],
    edges: np.ndarray,
    counts: np.ndarray,
    gap: int,
    m_limits: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
    """Fit a Gaussian curve to selected data based on provided parameters.

    Args:
    - dim (int): The dimension of the data.
    - max_ind (int): Index of the maximum value in the histogram.
    - minima (list[int]): List of indices representing the minimum points.
    - edges (np.ndarray): Array containing the bin edges of the histogram.
    - counts (np.ndarray): Array containing histogram counts.
    - gap (int): Minimum allowed gap size for fitting intervals.
    - m_limits (list[list[int]]): List of min and max limits for each
        dimension.

    Returns:
    - tuple[int, int, list[float]]: A tuple containing:
        - flag (int): Flag indicating the success (1) or failure (0) of
            the fitting process.
        - goodness (int): Goodness value representing the fitting quality
            (higher is better).
        - popt (list[float]): Optimal values for the parameters
            (mu, sigma, area) of the fitted Gaussian.

    This function attempts to fit a Gaussian curve to selected data within the
    specified dimension 'dim' based on provided histogram data ('edges' and
    'counts'). It uses 'max_ind' to initialize parameters and 'minima' to
    define the fitting interval.
    """

    # Initialize flag and goodness variables
    flag = 1
    goodness = 5

    # Extract relevant data within the specified minima
    edges_selection = edges[minima[2 * dim] : minima[2 * dim + 1]]
    all_axes = tuple(i for i in range(counts.ndim) if i != dim)
    counts_selection = np.sum(counts, axis=all_axes)
    counts_selection = counts_selection[minima[2 * dim] : minima[2 * dim + 1]]

    # Initial parameter guesses
    mu0 = edges[max_ind]
    sigma0 = (edges[minima[2 * dim + 1]] - edges[minima[2 * dim]]) / 2
    area0 = max(counts_selection) * np.sqrt(np.pi) * sigma0

    try:
        # Attempt to fit a Gaussian using curve_fit
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
    """Relabel states and update the state list based on occurrence in
        'all_the_labels'.

    Args:
    - all_the_labels (np.ndarray): Array containing labels assigned
        to each window in the trajectory.
    - states_list (list[StateUni]): List of StateUni objects representing
        different states.

    Returns:
    - tuple[np.ndarray, list[StateUni]]: A tuple containing:
        - all_the_labels (np.ndarray): Updated labels array with relabeled
            states.
        - relevant_states (list[StateUni]): Updated list of non-empty states,
        ordered by mean values.

    This function performs several operations to relabel the states and update
    the state list. It removes empty states, reorders them based on the mean
    values and relabels the labels in 'all_the_labels'.
    """
    # Step 1: Remove states with zero relevance
    relevant_states = [state for state in states_list if state.perc != 0.0]

    # Step 2: Sort states according to their mean value
    relevant_states.sort(key=lambda x: x.mean)

    # Step 3: Create a dictionary to map old state labels to new ones
    state_mapping = {
        state_index: index + 1
        for index, state_index in enumerate(
            [states_list.index(state) for state in relevant_states]
        )
    }

    relabel_map = np.zeros(len(states_list) + 1, dtype=int)
    for key, value in state_mapping.items():
        # Increment key by 1 to account for zero-indexed relabeling
        relabel_map[key + 1] = value

    # Step 4: Relabel the data in all_the_labels according to the
    # new states_list
    mask = all_the_labels != 0  # Create a mask for non-zero elements
    all_the_labels[mask] = relabel_map[all_the_labels[mask]]

    return all_the_labels, relevant_states


def find_intersection(st_0: StateUni, st_1: StateUni) -> Tuple[float, int]:
    """
    Finds the intersection between two Gaussians.

    Args:
    - st_0, st_1 (StateUni): the two states we are computing the threshold
        between

    Returns:
    - th_val (float): the value of the threshold
    - th_type (int): the type of the threshold (1 or 2)

    If the intersection exists, the threshold is type 1. If there are 2
    intersections, the one with higher value is chosen.
    If no intersection exists, the threshold is type 2. Its value will be
    the average between the two means, weighted with the two sigmas.
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

    Args:
    - list_of_states (list[StateUni]): List of StateUni objects representing
        potential states.
    - all_the_labels (np.ndarray): 2D NumPy array containing labels for each
        data point.
    - m_range (list[float]): Range of values in the data.

    Returns:
    - tuple: A tuple containing the final list of states
    (list[StateUni]) and the newly labeled data (np.ndarray).
    """
    ### Step 1: Merge together the strongly overlapping states
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

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    # Step 2: Calculate the final threshold values
    # and their types based on the intercept between neighboring states.

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

    # if updated_states[0].th_sup[0] < m_range[0]:
    #     updated_states.pop(0)
    #     updated_states[0].th_inf[0] = m_range[0]
    #     mask = all_the_labels > 1
    #     all_the_labels[mask] -= 1

    # if updated_states[-1].th_inf[0] > m_range[1]:
    #     updated_states.pop(-1)
    #     updated_states[-1].th_inf[1] = m_range[1]
    #     mask = all_the_labels == np.max(all_the_labels)
    #     all_the_labels[mask] -= 1

    # Step 3: Write the final states and final thresholds to text files.
    # The data is saved in two separate files:
    # 'final_states.txt' and 'final_thresholds.txt'.
    with open("final_states.txt", "w", encoding="utf-8") as file:
        print("# Mu \t Sigma \t A \t state_fraction", file=file)
        for state in updated_states:
            print(state.mean, state.sigma, state.area, state.perc, file=file)
    with open("final_thresholds.txt", "w", encoding="utf-8") as file:
        for state in updated_states:
            print(state.th_inf[0], state.th_inf[1], file=file)
        print(
            updated_states[-1].th_sup[0],
            updated_states[-1].th_sup[1],
            file=file,
        )

    # Step 5: Return the 'updated_states' as the output of the function.
    return updated_states, all_the_labels


def relabel_states_2d(
    all_the_labels: np.ndarray, states_list: list[StateMulti]
) -> Tuple[np.ndarray, List[StateMulti]]:
    """
    Reorders labels and merges strongly overlapping states in a
    multidimensional space.

    Args:
    - all_the_labels (np.ndarray): An ndarray containing labels associated
        with each state.
    - states_list (list[StateMulti]): A list of StateMulti objects
        representing states to be evaluated.

    Returns:
    - tuple[np.ndarray, list[StateMulti]]: A tuple containing:
        - Updated ndarray of labels reflecting the changes in state indices.
        - Modified list of StateMulti objects after merging and relabeling
        states.
    """

    ### Step 1: Remove states with zero relevance
    sorted_states = [state for state in states_list if state.perc != 0.0]

    ### Step 2: Sort states according to their relevance
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

    ### Step 3: Merge together the states which are strongly overlapping
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

    ### Step 4: print informations on the final states
    with open("final_states.txt", "w", encoding="utf-8") as file:
        print("#center_coords, semiaxis, fraction_of_data", file=file)
        for state in updated_states:
            centers = "[" + str(state.mean[0]) + ", "
            for tmp in state.mean[1:-1]:
                centers += str(tmp) + ", "
            centers += str(state.mean[-1]) + "]"
            axis = "[" + str(state.axis[0]) + ", "
            for tmp in state.axis[1:-1]:
                axis += str(tmp) + ", "
            axis += str(state.axis[-1]) + "]"
            print(centers, axis, state.perc, file=file)

    return all_the_labels, updated_states
