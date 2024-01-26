"""
Should contains all the functions in common between the 2 codes.
"""
import os
import numpy as np
import plotly.graph_objects as go
import scipy.optimize
import scipy.signal
from onion_clustering.classes import *

def read_input_data():
    """
    Attempt to read the content of 'data_directory.txt' file
    and load it into a NumPy array as strings.
    """
    try:
        data_dir = np.loadtxt('data_directory.txt', dtype=str)
    except:
        print('\tdata_directory.txt file missing or wrongly formatted.')

    print('* Reading data from', data_dir)

    if data_dir.size == 1:
        return str(data_dir)
    return data_dir

def moving_average(data: np.ndarray, window: int):
    """Applies a moving average filter to a 1D or 2D NumPy array.

    Args:
    - data (np.ndarray): The input array to be smoothed.
    - window (int): The size of the moving average window.

    Returns:
    - np.ndarray: The smoothed array obtained after applying the moving average filter.

    Raises:
    - ValueError: If the input array dimension is not supported
    (only 1D and 2D arrays are supported).
    """

    # Step 1: Create a NumPy array 'weights' with the values 1.0 repeated 'window' times.
    # Then, divide each element of 'weights' by the 'window' value to get the average weights.
    weights = np.ones(window) / window

    # Step 2: Apply the moving average filter to the 'data' array using the 'weights' array.
    # The 'np.convolve' function performs a linear convolution between 'data' and 'weights'.
    # The result is a smoothed version of the 'data',
    # where each point represents the weighted average of its neighbors.
    if data.ndim == 1:
        return np.convolve(data, weights, mode='valid')
    if data.ndim >= 2:
        return np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'),
            axis=1, arr=data)
    raise ValueError('Invalid array dimension. Only 1D and 2D arrays are supported.')

def moving_average_2d(data: np.ndarray, side: int):
    """Applies a 2D moving average filter to a NumPy array.

    Args:
    - data (np.ndarray): The 2D input array to be smoothed.
    - side (int): The side length of the square moving average window (must be an odd number).

    Returns:
    - np.ndarray: The smoothed array obtained after applying the 2D moving average filter.

    Raises:
    - ValueError: If the side length 'side' is not an odd number.
    """

    if side % 2 == 0:                           # Check if side is an odd number
        raise ValueError("L must be an odd number.")
    half_width = (side - 1) // 2                # Calculate the half-width of the moving window
    result = np.zeros_like(data, dtype=float)   # Initialize the result array with zeros

    for index in np.ndindex(*data.shape):
        slices = tuple(slice(max(0, i - half_width), min(data.shape[dim], i + half_width + 1))
            for dim, i in enumerate(index))
        subarray = data[slices]
        # Calculate the average if the subarray is not empty
        if subarray.size > 0:
            result[index] = subarray.mean()

    return result

def dense_interpolation(m_clean: np.ndarray, dense_factor: int):
    """
    This is work in progress. Function for dense interpolation:
    dense_factor points are added, via linear interpolation, between
    consecutive data points.
    """
    m_dense = []
    for _, data_i in enumerate(m_clean):
        tmp = []
        for j, _ in enumerate(data_i[1:]):
            for k in range(dense_factor):
                new_point = data_i[j] + (data_i[j + 1] - data_i[j])/dense_factor*k
                tmp.append(new_point)
        m_dense.append(tmp)
    m_dense_arr = np.array(m_dense)
    return m_dense_arr

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
    ax.set_xlabel(r'Normalized signal')
    ax.set_ylabel(r'Probability distribution')

def param_grid(par: Parameters, trj_len: int):
    """Generates parameter grids for tau_window and t_smooth.

    Args:
    - par (Parameters): An instance of the Parameters class containing parameter details.
    - trj_len (int): Length of the trajectory data.

    Returns:
    - tau_window (List[int]): A list of tau_window values.
    - t_smooth (List[int]): A list of t_smooth values.

    This function generates grids of values for 'tau_window' and 't_smooth' based on
    the provided 'Parameters' instance and the length of the trajectory. It calculates
    the values for 'tau_window' within the range defined in 'Parameters' and generates
    't_smooth' values within the specified range.
    """

    if par.max_tau_w == -1:
        par.max_tau_w = trj_len - par.max_t_smooth
    tmp = np.geomspace(par.min_tau_w, par.max_tau_w, num=par.num_tau_w, dtype=int)
    tau_window = []
    for tau_w in tmp:
        if tau_w not in tau_window:
            tau_window.append(tau_w)
    print('* Tau_w used:', tau_window)

    t_smooth = list(range(par.min_t_smooth, par.max_t_smooth + 1, par.step_t_smooth))
    print('* t_smooth used:', t_smooth)

    return tau_window, t_smooth

def gaussian(x_points: np.ndarray, x_mean: float, sigma: float, area: float):
    """Compute the Gaussian function values at given points 'x'.

    Args:
    - x_points (np.ndarray): Array of input values.
    - x_mean (float): Mean value of the Gaussian function.
    - sigma (float): Standard deviation of the Gaussian function.
    - area (float): Area under the Gaussian curve.

    Returns:
    - np.ndarray: Gaussian function values computed at the input points 'x_points'.

    This function calculates the values of a Gaussian function at the given
    array of points 'x_points' using the provided 'x_mean', 'sigma' (standard deviation),
    and 'area' (area under the curve) parameters. It returns an array of Gaussian
    function values corresponding to the input 'x_points'.
    """

    return np.exp(-((x_points - x_mean)/sigma)**2)*area/(np.sqrt(np.pi)*sigma)

def custom_fit(dim: int, max_ind: int, minima: list[int],
    edges: np.ndarray, counts: np.ndarray, gap: int, m_limits: np.ndarray):
    """Fit a Gaussian curve to selected data based on provided parameters.

    Args:
    - dim (int): The dimension to fit the Gaussian curve (0 for x, 1 for y, etc.).
    - max_ind (int): Index of the maximum value in the histogram.
    - minima (list[int]): List of indices representing the minimum points.
    - edges (np.ndarray): Array containing the bin edges of the histogram.
    - counts (np.ndarray): Array containing histogram counts.
    - gap (int): Minimum allowed gap size for fitting intervals.
    - m_limits (list[list[int]]): List of min and max limits for each dimension.

    Returns:
    - tuple[int, int, list[float]]: A tuple containing:
        - flag (int): Flag indicating the success (1) or failure (0) of the fitting process.
        - goodness (int): Goodness value representing the fitting quality (higher is better).
        - popt (list[float]): Optimal values for the parameters
        (mu, sigma, area) of the fitted Gaussian.

    This function attempts to fit a Gaussian curve to selected data within the specified
    dimension 'dim' based on provided histogram data ('edges' and 'counts'). It uses
    'max_ind' to initialize parameters and 'minima' to define the fitting interval.

    'gap' represents the minimum allowed gap size for fitting intervals.
    'm_limits' contains minimum and maximum limits for each dimension.

    It returns a tuple consisting of:
    - 'flag' indicating success (1) or failure (0) of the fitting process.
    - 'goodness', a value representing the quality of the fit (higher is better).
    - 'popt', the optimal values for the parameters (mu, sigma, area) of the fitted Gaussian curve.
    """

    # Initialize flag and goodness variables
    flag = 1
    goodness = 5

    # Extract relevant data within the specified minima
    edges_selection = edges[minima[2*dim]:minima[2*dim + 1]]
    all_axes = tuple(i for i in range(counts.ndim) if i != dim)
    counts_selection = np.sum(counts, axis=all_axes)
    counts_selection = counts_selection[minima[2*dim]:minima[2*dim + 1]]

    # Initial parameter guesses
    mu0 = edges[max_ind]
    sigma0 = (edges[minima[2*dim + 1]] - edges[minima[2*dim]])/2
    area0 = max(counts_selection)*np.sqrt(np.pi)*sigma0

    try:
        # Attempt to fit a Gaussian using curve_fit
        popt, pcov = scipy.optimize.curve_fit(gaussian, edges_selection, counts_selection,
            p0=[mu0, sigma0, area0], bounds=([m_limits[dim][0], 0.0, 0.0],
            [m_limits[dim][1], np.inf, np.inf]))

        # Check goodness of fit and update the goodness variable
        if popt[0] < edges_selection[0] or popt[0] > edges_selection[-1]:
            goodness -= 1
        if popt[1] > edges_selection[-1] - edges_selection[0]:
            goodness -= 1
        if popt[2] < area0/2:
            goodness -= 1

        # Calculate parameter errors
        perr = np.sqrt(np.diag(pcov))
        for j, par_err in enumerate(perr):
            if par_err/popt[j] > 0.5:
                goodness -= 1

        # Check if the fitting interval is too small in either dimension
        if minima[2*dim + 1] - minima[2*dim] <= gap:
            goodness -= 1

    except RuntimeError:
        print('\tFit: Runtime error. ')
        flag = 0
        popt = []
    except TypeError:
        print('\tFit: TypeError.')
        flag = 0
        popt = []
    except ValueError:
        print('\tFit: ValueError.')
        flag = 0
        popt = []
    return flag, goodness, popt

def relabel_states(all_the_labels: np.ndarray, states_list: list[StateUni]):
    """Relabel states and update the state list based on occurrence in 'all_the_labels'.

    Args:
    - all_the_labels (np.ndarray): Array containing labels assigned
        to each window in the trajectory.
    - states_list (list[StateUni]): List of StateUni objects representing different states.

    Returns:
    - tuple[np.ndarray, list[StateUni]]: A tuple containing:
        - all_the_labels (np.ndarray): Updated labels array with relabeled states.
        - relevant_states (list[StateUni]): Updated list of non-empty states,
        ordered by mean values.

    This function performs several operations to relabel the states and update the state list.
    It removes empty states, reorders them based on the mean values and relabels the labels in
    'all_the_labels'.
    """
    # Step 1: Remove states with zero relevance
    relevant_states = [state for state in states_list if state.perc != 0.0]

    # Step 2: Sort states according to their mean value
    relevant_states.sort(key=lambda x: x.mean)

    # Step 3: Create a dictionary to map old state labels to new ones
    state_mapping = {state_index: index + 1 for index, state_index
        in enumerate([states_list.index(state) for state in relevant_states])}

    relabel_map = np.zeros(len(states_list) + 1, dtype=int)
    for key, value in state_mapping.items():
        relabel_map[key + 1] = value  # Increment key by 1 to account for zero-indexed relabeling

    # Step 4: Relabel the data in all_the_labels according to the new states_list
    mask = all_the_labels != 0  # Create a mask for non-zero elements
    all_the_labels[mask] = relabel_map[all_the_labels[mask]]

    return all_the_labels, relevant_states

def find_intersection(st_0: StateUni, st_1: StateUni):
    """
    Finds the intersection between two Gaussians.

    Args:
    - st_0, st_1 (StateUni): the two states we are computing the threshold between

    Returns:
    - th_val (float): the value of the threshold
    - th_type (int): the type of the threshold (1 or 2)

    If the intersection exists, the threshold is type 1. If there are 2 intersections,
    the one with higher value is chosen.
    If no intersection exists, the threshold is type 2. Its value will be the average
    between the two means, weighted with the two sigmas.
    """
    coeff_a = st_1.sigma**2 - st_0.sigma**2
    coeff_b = -2*(st_0.mean*st_1.sigma**2 - st_1.mean*st_0.sigma**2)
    tmp_c = np.log(st_0.area*st_1.sigma/st_1.area/st_0.sigma)
    coeff_c = ((st_0.mean*st_1.sigma)**2
        - (st_1.mean*st_0.sigma)**2
        - ((st_0.sigma*st_1.sigma)**2)*tmp_c)
    delta = coeff_b**2 - 4*coeff_a*coeff_c
    if coeff_a == 0.0:
        only_th = ((st_0.mean + st_1.mean)/2
            - st_0.sigma**2/2/(st_1.mean - st_0.mean)*np.log(st_0.area/st_1.area))
        return only_th, 1
    if delta >= 0:
        th_plus = (- coeff_b + np.sqrt(delta))/(2*coeff_a)
        th_minus = (- coeff_b - np.sqrt(delta))/(2*coeff_a)
        intercept_plus = gaussian(th_plus, st_0.mean, st_0.sigma, st_0.area)
        intercept_minus = gaussian(th_minus, st_0.mean, st_0.sigma, st_0.area)
        if intercept_plus >= intercept_minus:
            return th_plus, 1
        return th_minus, 1
    th_aver = (st_0.mean/st_0.sigma + st_1.mean/st_1.sigma)/(1/st_0.sigma + 1/st_1.sigma)
    return th_aver, 2

def set_final_states(list_of_states: list[StateUni], all_the_labels: np.ndarray,
    m_range: list[float]):
    """
    Assigns final states and relabels labels based on specific criteria.

    Args:
    - list_of_states (list[StateUni]): List of StateUni objects representing potential states.
    - all_the_labels (np.ndarray): 2D NumPy array containing labels for each data point.
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
                if st_0.peak > st_1.peak and abs(st_1.mean - st_0.mean) < st_0.sigma:
                    proposed_merge.append([j, i])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([ pair[0] for pair in proposed_merge ])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            list_of_distances = [np.linalg.norm(list_of_states[pair[1]].mean -
                list_of_states[pair[0]].mean) for pair in candidate_merge]
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

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(all_the_labels.shape)

    final_map = np.zeros(max(np.unique(all_the_labels)) + 1, dtype=int)
    for i, el in enumerate(np.unique(all_the_labels)):
        final_map[el] = i
    for i, particle in enumerate(all_the_labels):
        for j, el in enumerate(particle):
            all_the_labels[i][j] = final_map[el]

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [state for i, state in enumerate(list_of_states) if i not in states_to_remove]

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    # Step 2: Calculate the final threshold values
    # and their types based on the intercept between neighboring states.

    updated_states[0].th_inf[0] = m_range[0]
    updated_states[0].th_inf[1] = 0

    for i in range(len(updated_states) - 1):
        th_val, th_type = find_intersection(updated_states[i], updated_states[i + 1])
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
        mask = all_the_labels == max(all_the_labels)
        all_the_labels[mask] -= 1

    # Step 3: Write the final states and final thresholds to text files.
    # The data is saved in two separate files: 'final_states.txt' and 'final_thresholds.txt'.
    with open('final_states.txt', 'w', encoding="utf-8") as file:
        print('# Mu \t Sigma \t A \t state_fraction', file=file)
        for state in updated_states:
            print(state.mean, state.sigma, state.area, state.perc, file=file)
    with open('final_thresholds.txt', 'w', encoding="utf-8") as file:
        for state in updated_states:
            print(state.th_inf[0], state.th_inf[1], file=file)
        print(updated_states[-1].th_sup[0], updated_states[-1].th_sup[1], file=file)

    # Step 5: Return the 'updated_states' as the output of the function.
    return updated_states, all_the_labels

def relabel_states_2d(all_the_labels: np.ndarray, states_list: list[StateMulti]):
    """
    Reorders labels and merges strongly overlapping states in a multidimensional space.

    Args:
    - all_the_labels (np.ndarray): An ndarray containing labels associated with each state.
    - states_list (list[StateMulti]): A list of StateMulti objects
        representing states to be evaluated.

    Returns:
    - tuple[np.ndarray, list[StateMulti]]: A tuple containing:
        - Updated ndarray of labels reflecting the changes in state indices.
        - Modified list of StateMulti objects after merging and relabeling states.
    """

    ### Step 1: Remove states with zero relevance
    sorted_states = [state for state in states_list if state.perc != 0.0]

    ### Step 2: Sort states according to their relevance
    # Create a dictionary to map old state labels to new ones
    state_mapping = {index: i + 1 for i, index
        in enumerate(np.argsort([-state.perc for state in sorted_states]))}

    sorted_states.sort(key=lambda x: x.perc, reverse=True)

    # Relabel the data in all_the_labels according to the new states_list
    mask = all_the_labels != 0  # Create a mask for non-zero elements
    all_the_labels[mask] = np.vectorize(state_mapping.get,
        otypes=[int])(all_the_labels[mask] - 1, 0)

    ### Step 3: Merge together the states which are strongly overlapping
    # Find all the possible merges
    proposed_merge = []
    for i, st_0 in enumerate(sorted_states):
        for j, st_1 in enumerate(sorted_states):
            if j > i:
                diff = np.abs(np.subtract(st_1.mean, st_0.mean))
                if np.all(diff < [ max(st_0.sigma[k], st_1.sigma[k]) for k in range(diff.size) ]):
                    proposed_merge.append([j, i])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([ pair[0] for pair in proposed_merge ])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            list_of_distances = [np.linalg.norm(sorted_states[pair[1]].mean -
                sorted_states[pair[0]].mean) for pair in candidate_merge]
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

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(all_the_labels.shape)

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [state for i, state in enumerate(sorted_states) if i not in states_to_remove]

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    ### Step 4: print informations on the final states
    with open('final_states.txt', 'w', encoding="utf-8") as file:
        print('#center_coords, semiaxis, fraction_of_data', file=file)
        for state in updated_states:
            centers = '[' + str(state.mean[0]) + ', '
            for tmp in state.mean[1:-1]:
                centers += str(tmp) + ', '
            centers += str(state.mean[-1]) + ']'
            axis = '[' + str(state.axis[0]) + ', '
            for tmp in state.axis[1:-1]:
                axis += str(tmp) + ', '
            axis += str(state.axis[-1]) + ']'
            print(centers, axis, state.perc, file=file)

    return all_the_labels, updated_states

def assign_single_frames(all_the_labels: np.ndarray, tau_window: int):
    """
    Assigns labels to individual frames by repeating the existing labels.

    Args:
    - all_the_labels (np.ndarray): An ndarray containing labels associated with each state.
    - tau_window (int): The number of frames for which the labels are to be assigned.

    Returns:
    - np.ndarray: An updated ndarray with labels assigned to individual frames
        by repeating the existing labels.
    """
    print('* Assigning labels to the single frames...')
    new_labels = np.repeat(all_the_labels, tau_window, axis=1)
    return new_labels

def plot_tra_figure(number_of_states: np.ndarray, fraction_0: np.ndarray,
    par: Parameters):
    """
    Plots time resolution analysis figures based on the number of states
    and fraction of a specific state.

    Args:
    - number_of_states (np.ndarray): Array containing the number of states.
    - fraction_0 (np.ndarray): Array containing the fraction of a specific state.
    - par (Parameters): Instance of Parameters with time-related parameters.
    """
    t_conv, units = par.t_conv, par.t_units
    min_t_s, max_t_s, step_t_s = par.min_t_smooth, par.max_t_smooth, par.step_t_smooth

    time = number_of_states.T[0]*t_conv
    number_of_states = number_of_states[:, 1:].T
    fraction_0 = fraction_0[:, 1:].T

    for i, t_smooth in enumerate(range(min_t_s, max_t_s + 1, step_t_s)):
        fig, ax = plt.subplots()
        y_signal = number_of_states[i]
        y_2 = fraction_0[i]

        ### General plot settings ###
        ax.plot(time, y_signal, marker='o')
        ax.set_xlabel(r'Time resolution $\Delta t$ ' + units)#, weight='bold')
        ax.set_ylabel(r'# environments', weight='bold', c='#1f77b4')
        ax.set_xscale('log')
        ax.set_xlim(time[0]*0.75, time[-1]*1.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ### Top x-axes settings ###
        ax2 = ax.twiny()
        ax2.set_xlabel(r'Time resolution $\Delta t$ [frames]')
        ax2.set_xscale('log')
        ax2.set_xlim(time[0]*0.75/t_conv, time[-1]*1.5/t_conv)

        axr = ax.twinx()
        axr.plot(time, y_2, marker='o', c='#ff7f0e')
        axr.set_ylabel('Population of env 0', weight='bold', c='#ff7f0e')

        fig.savefig('output_figures/Time_resolution_analysis_' + str(t_smooth) + '.png', dpi=600)

def sankey(all_the_labels: np.ndarray, tmp_frame_list: list[int],
    par: Parameters, filename: str):
    """
    Computes and plots a Sankey diagram based on provided data and parameters.

    Args:
    - all_the_labels (np.ndarray): Array containing labels for each frame.
    - tmp_frame_list (list[int]): List of frame indices.
    - par (Parameters): Instance of Parameters containing time-related parameters.
    - filename (str): Name of the file for the output Sankey diagram.

    Steps:
    - Computes transition matrices for each time window based on label data.
    - Constructs source, target, and value arrays for the Sankey diagram.
    - Generates node labels and color palette for the diagram visualization.
    - Creates Sankey diagram using Plotly library and custom node/link data.
    - Saves the generated Sankey diagram as an image file.
    """
    print('* Computing and plotting the Sankey diagram...')

    # Determine the number of unique states in the data.
    frame_list = np.array(tmp_frame_list)

    unique_labels = np.unique(all_the_labels)
    # If there are no assigned window, we still need the "0" state
    # for consistency:
    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)
    n_states = unique_labels.size

    # Create arrays to store the source, target, and value data for the Sankey diagram.
    source = np.empty((frame_list.size - 1) * n_states**2)
    target = np.empty((frame_list.size - 1) * n_states**2)
    value = np.empty((frame_list.size - 1) * n_states**2)

    # Initialize a counter variable.
    count = 0

    # Create temporary lists to store node labels for the Sankey diagram.
    tmp_label1 = []
    tmp_label2 = []

    # Loop through the frame_list and calculate the transition matrix for each time window.
    for i, t_0 in enumerate(frame_list[:-1]):
        # Calculate the time jump for the current time window.
        t_jump = frame_list[i + 1] - frame_list[i]

        # Initialize a matrix to store the transition counts between states.
        trans_mat = np.zeros((n_states, n_states))

        # Iterate through the current time window and increment the transition counts in trans_mat.
        for label in all_the_labels:
            trans_mat[label[t_0]][label[t_0 + t_jump]] += 1

        # Store the source, target, and value for the Sankey diagram based on trans_mat.
        for j, row in enumerate(trans_mat):
            for k, elem in enumerate(row):
                source[count] = j + i * n_states
                target[count] = k + (i + 1) * n_states
                value[count] = elem
                count += 1

        # Calculate the starting and ending fractions for each state and store node labels.
        for j in range(n_states):
            start_fr = np.sum(trans_mat[j]) / np.sum(trans_mat)
            end_fr = np.sum(trans_mat.T[j]) / np.sum(trans_mat)
            if i == 0:
                tmp_label1.append('State ' + str(j) + ': ' + "{:.2f}".format(start_fr * 100) + '%')
            tmp_label2.append('State ' + str(j) + ': ' + "{:.2f}".format(end_fr * 100) + '%')

    # Concatenate the temporary labels to create the final node labels.
    label = np.concatenate((tmp_label1, np.array(tmp_label2).flatten()))

    # Generate a color palette for the Sankey diagram.
    palette = []
    cmap = plt.get_cmap('viridis', n_states)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Tile the color palette to match the number of frames.
    color = np.tile(palette, frame_list.size)

    # Create dictionaries to define the Sankey diagram nodes and links.
    node = dict(label=label, pad=30, thickness=20, color=color)
    link = dict(source=source, target=target, value=value)

    # Create the Sankey diagram using Plotly.
    sankey_data = go.Sankey(link=link, node=node, arrangement="perpendicular")
    fig = go.Figure(sankey_data)

    # Add the title with the time information.
    fig.update_layout(title='Frames: ' + str(frame_list * par.t_conv) + ' ' + par.t_units)

    fig.write_image('output_figures/' + filename + '.png', scale=5.0)

def plot_state_populations(all_the_labels: np.ndarray,
    par: Parameters, filename: str):
    """
    Plots the populations of states over time.

    Args:
    - all_the_labels (np.ndarray): Array containing state labels for each time step.
    - par (Parameters): Instance of Parameters class containing time-related information.
    - filename (str): Name of the file to save the generated plot.

    Steps:
    - Computes the populations of each state at different time steps.
    - Creates a plot illustrating state populations against time.
    - Utilizes Matplotlib to generate the plot based on provided data.
    - Saves the resulting plot as an image file.

    Note:
    - Uses Matplotlib for creating the state population vs. time visualization.
    - Provides options for file naming and displaying the plot based on the parameters.
    """
    print('* Printing populations vs time...')
    num_part = all_the_labels.shape[0]

    unique_labels = np.unique(all_the_labels)
    # If there are no assigned window, we still need the "0" state
    # for consistency:
    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)

    list_of_populations = []
    for label in unique_labels:
        population = np.sum(all_the_labels == label, axis=0)
        list_of_populations.append(population / num_part)

    # Generate the color palette.
    palette = []
    n_states = unique_labels.size
    cmap = plt.get_cmap('viridis', n_states)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    fig, ax = plt.subplots()
    t_steps = all_the_labels.shape[1]
    time = par.print_time(t_steps)
    for label, pop in enumerate(list_of_populations):
        # pop_full = np.repeat(pop, par.tau_w)
        ax.plot(time, pop, label='ENV' + str(label), color=palette[label])
    ax.set_xlabel(r'Time ' + par.t_units)
    ax.set_ylabel(r'Population')
    ax.legend()

    fig.savefig('output_figures/' + filename + '.png', dpi=600)

def print_mol_labels_fbf_gro(all_the_labels: np.ndarray):
    """
    Prints color IDs for Ovito visualization in GRO format.

    Args:
    - all_the_labels (np.ndarray): Array containing molecular labels for each frame.

    Steps:
    - Creates a file ('all_cluster_IDs_gro.dat') to store color IDs for Ovito visualization.
    - Iterates through each frame's molecular labels and writes them to the file in GRO format.
    """
    print('* Print color IDs for Ovito...')
    with open('all_cluster_IDs_gro.dat', 'w', encoding="utf-8") as file:
        for labels in all_the_labels:
            # Join the elements of 'labels' using a space as the separator and write to the file.
            print(' '.join(map(str, labels)), file=file)

def print_signal_with_labels(m_clean: np.ndarray, all_the_labels: np.ndarray):
    """
    Creates a file ('signal_with_labels.dat') with signal values and associated cluster labels.

    Args:
    - m_clean (np.ndarray): Signal array containing the signals for each frame.
    - all_the_labels (np.ndarray): Array containing cluster labels for each frame.

    Steps:
    - Checks the dimensionality of 'm_clean' to determine the signal attributes.
    - Writes the signals along with cluster labels for each frame to 'signal_with_labels.dat'.
    - Assumes the structure of 'm_clean' with signal values based on dimensionality (2D or 3D).
    - Incorporates 'all_the_labels' as cluster labels for respective frames.
    """
    with open('signal_with_labels.dat', 'w+', encoding="utf-8") as file:
        if m_clean.shape[2] == 2:
            print("Signal 1 Signal 2 Cluster Frame", file=file)
        else:
            print("Signal 1 Signal 2 Signal 3 Cluster Frame", file=file)
        for j in range(all_the_labels.shape[1]):
            for i in range(all_the_labels.shape[0]):
                if m_clean.shape[2] == 2:
                    print(m_clean[i][j][0], m_clean[i][j][1],
                        all_the_labels[i][j], j + 1, file=file)
                else:
                    print(m_clean[i][j][0], m_clean[i][j][1], m_clean[i][j][2],
                        all_the_labels[i][j], j + 1, file=file)

def print_mol_labels_fbf_xyz(all_the_labels: np.ndarray):
    """
    Prints color IDs for Ovito visualization in XYZ format.

    Args:
    - all_the_labels (np.ndarray): Array containing molecular labels for each frame.

    Steps:
    - Creates a file ('all_cluster_IDs_xyz.dat') to store color IDs for Ovito visualization.
    - Iterates through each frame's molecular labels and writes them to the file in XYZ format.
    """
    print('* Print color IDs for Ovito...')
    with open('all_cluster_IDs_xyz.dat', 'w+', encoding="utf-8") as file:
        for j in range(all_the_labels.shape[1]):
            # Print two lines containing '#' to separate time steps.
            print('#', file=file)
            print('#', file=file)
            # Use np.savetxt to write the labels for each time step efficiently.
            np.savetxt(file, all_the_labels[:, j], fmt='%d', comments='')

def print_mol_labels_fbf_lam(all_the_labels: np.ndarray):
    """
    Prints color IDs for Ovito visualization in .lammps format.

    Args:
    - all_the_labels (np.ndarray): Array containing molecular labels for each frame.

    Steps:
    - Creates a file ('all_cluster_IDs_lam.dat') to store color IDs for Ovito visualization.
    - Iterates through each frame's molecular labels and writes them to the file in .lammps format.
    """
    print('* Print color IDs for Ovito...')
    with open('all_cluster_IDs_lam.dat', 'w', encoding="utf-8") as file:
        for j in range(all_the_labels.shape[1]):
            # Print nine lines containing '#' to separate time steps.
            for _ in range(9):
                print('#', file=file)
            # Use np.savetxt to write the labels for each time step efficiently.
            np.savetxt(file, all_the_labels[:, j], fmt='%d', comments='')

def print_colored_trj_from_xyz(trj_file: str, all_the_labels: np.ndarray, par: Parameters):
    """
    Creates a new XYZ file ('colored_trj.xyz') by coloring the original trajectory
    based on cluster labels.

    Args:
    - trj_file (str): Path to the original XYZ trajectory file.
    - all_the_labels (np.ndarray): Array containing cluster labels for each frame.
    - par (Parameters): Object storing parameters for processing.

    Steps:
    - Reads the original trajectory file 'trj_file'.
    - Removes the initial and final frames based on 'par.t_smooth',
        'par.t_delay', and available frames.
    - Creates a new XYZ file 'colored_trj.xyz' by adding cluster labels to the particle entries.
    """
    if os.path.exists(trj_file):
        print('* Loading trajectory.xyz...')
        with open(trj_file, "r", encoding="utf-8") as in_file:
            tmp = [line.strip().split() for line in in_file]

        num_of_particles = all_the_labels.shape[0]
        total_time = all_the_labels.shape[1]
        nlines = (num_of_particles + 2) * total_time

        frames_to_remove = int(par.t_smooth/2) #+ par.t_delay
        print('\t Removing the first', frames_to_remove, 'frames...')
        tmp = tmp[frames_to_remove * (num_of_particles + 2):]

        frames_to_remove = int((len(tmp) - nlines)/(num_of_particles + 2))
        print('\t Removing the last', frames_to_remove, 'frames...')
        tmp = tmp[:nlines]

        with open('colored_trj.xyz', "w+", encoding="utf-8") as out_file:
            i = 0
            for j in range(total_time):
                print(tmp[i][0], file=out_file)
                print('Properties=species:S:1:pos:R:3', file=out_file)
                for k in range(num_of_particles):
                    print(all_the_labels[k][j],
                        tmp[i + 2 + k][1], tmp[i + 2 + k][2], tmp[i + 2 + k][3], file=out_file)
                i += num_of_particles + 2
    else:
        print('No ' + trj_file + ' found for coloring the trajectory.')
