"""
Should contains all the functions in common between the 2 codes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import scipy.optimize
import scipy.signal
from classes import *

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

def read_data(filename: str):
    """Reads data from supported file formats: .npz, .npy, or .txt.

    Args:
    - filename (str): The path to the file containing the data.

    Returns:
    - Optional[np.ndarray]: A NumPy array containing the loaded data if successful,
      otherwise returns None.

    Raises:
    - IOError: If the file format is unsupported or if there's an issue reading the file.
    """

    # Check if the filename ends with a supported format.
    if filename.endswith(('.npz', '.npy', '.txt')):
        try:
            if filename.endswith('.npz'):
                with np.load(filename) as data:
                    # Load the first variable (assumed to be the data) into a NumPy array.
                    data_name = data.files[0]
                    m_raw = np.array(data[data_name])
            elif filename.endswith('.npy'):
                m_raw = np.load(filename)
            else: # .txt file
                m_raw = np.loadtxt(filename)
            print('\tOriginal data shape:', m_raw.shape)
            return m_raw
        except Exception as exc_msg:
            print(f'\tERROR: Failed to read data from {filename}. Reason: {exc_msg}')
            return None
    else:
        print('\tERROR: unsupported format for input file.')
        return None

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

    t_smooth = list(range(par.min_t_smooth, par.max_t_smooth + 1))
    # t_smooth = [ ts for ts in range(par.min_t_smooth, par.max_t_smooth + 1) ]
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

def gaussian_2d(r_points: np.ndarray, x_mean: float, y_mean: float, sigmax: float, sigmay: float, area: float):
    """Compute the 2D Gaussian function values at given radial points 'r_points'.

    Args:
    - r_points (np.ndarray): Array of radial points in a 2D space.
    - x_mean (float): Mean value along the x-axis of the Gaussian function.
    - y_mean (float): Mean value along the y-axis of the Gaussian function.
    - sigmax (float): Standard deviation along the x-axis of the Gaussian function.
    - sigmay (float): Standard deviation along the y-axis of the Gaussian function.
    - area (float): Total area under the 2D Gaussian curve.

    Returns:
    - np.ndarray: 2D Gaussian function values computed at the radial points.

    This function calculates the values of a 2D Gaussian function at given radial points 'r_points'
    centered around the provided means ('x_mean' and 'y_mean') and standard deviations
    ('sigmax' and 'sigmay'). The 'area' parameter represents the total area
    under the 2D Gaussian curve. It returns an array of 2D Gaussian function values
    computed at the input radial points 'r_points'.
    """

    r_points[0] -= x_mean
    r_points[1] -= y_mean
    arg = (r_points[0]/sigmax)**2 + (r_points[1]/sigmay)**2
    normalization = np.pi*sigmax*sigmay
    gauss = np.exp(-arg)*area/normalization
    return gauss.ravel()

def custom_fit(dim: int, max_ind: int, minima: list[int], edges: np.ndarray, counts: np.ndarray, gap: int, m_limits: list[list[int]]):
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

def relabel_states(all_the_labels: np.ndarray, states_list: list[State]):
    """Relabel states and update the state list based on occurrence in 'all_the_labels'.

    Args:
    - all_the_labels (np.ndarray): Array containing labels assigned
        to each window in the trajectory.
    - states_list (list[State]): List of State objects representing different states.

    Returns:
    - tuple[np.ndarray, list[State]]: A tuple containing:
        - tmp2 (np.ndarray): Updated labels array with relabeled states.
        - list1 (list[State]): Updated list of non-empty states, ordered by mean values.

    This function performs several operations to relabel the states and update the state list
    based on their occurrence in 'all_the_labels'. It removes empty states, relabels the states
    in 'all_the_labels', and reorders 'list1' based on the mean values of the non-empty states.

    The relabeling process:
    1. Removes empty states (where 'perc' value is 0.0) from 'states_list'.
    2. Obtains unique labels from the 'all_the_labels' array.
    3. Relabels the states in 'all_the_labels' from 0 to n_states-1 based on their occurrence.
    4. Orders the states in 'list1' according to the mean values in ascending order.
    5. Creates 'tmp2' by relabeling the states in 'all_the_labels' based on the sorted order.

    'tmp2' is the updated labels array with relabeled states.
    'list1' is the updated list of non-empty states ordered by their mean values.
    """

    # Step 1: Remove empty states from the 'list_of_states' and keep only non-empty states.
    # A non-empty state is one where the third element (index 2) is not equal to 0.0.
    list1 = [state for state in states_list if state.perc != 0.0]

    # Step 2: Get the unique labels from the 'all_the_labels' array.
    list_unique = np.unique(all_the_labels)

    # Step 3: Relabel the states from 0 to n_states-1 based on their occurrence in 'all_the_labels'.
    # Create a dictionary to map unique labels to new indices.
    label_to_index = {label: index for index, label in enumerate(list_unique)}
    # Use vectorized indexing to relabel the states in 'all_the_labels'.
    tmp1 = np.vectorize(label_to_index.get)(all_the_labels)

    for mol_id in range(tmp1.shape[0]):
        for time_id in range(tmp1.shape[1]):
            tmp1[mol_id][time_id] = list_unique[tmp1[mol_id][time_id]]

    # Step 4: Order the states according to the mu values in the 'list1' array.
    list1.sort(key=lambda state: state.mean)

    # Create 'tmp2' by relabeling the states based on the sorted order.
    tmp2 = np.zeros_like(tmp1)
    for old_label in list_unique:
        tmp2[tmp1 == old_label] = label_to_index.get(old_label)

    return tmp2, list1

def set_final_states(list_of_states: list[State], all_the_labels: np.ndarray, m_range: list[float]):
    """
    Assigns final states and relabels labels based on specific criteria.

    Args:
    - list_of_states (list[State]): List of State objects representing potential states.
    - all_the_labels (np.ndarray): 2D NumPy array containing labels for each data point.
    - m_range (list[float]): Range of values in the data.

    Returns:
    - tuple: A tuple containing the final list of states (list[State]) and
             the newly labeled data (np.ndarray).
    """

    # Step 1: Define a criterion to determine which states are considered "final."
    # Iterate over pairs of states to compare their properties.
    old_to_new_map = []
    tmp_list = []

    for id_s0, st_0 in enumerate(list_of_states):
        for id_s1, st_1 in enumerate(list_of_states[id_s0 + 1:]):
            # Check whether the criteria for considering a state as "final" is met.
            if st_0.peak > st_1.peak and abs(st_1.mean - st_0.mean) < st_0.sigma:
                tmp_list.append(id_s1 + id_s0 + 1)
                old_to_new_map.append([id_s1 + id_s0 + 1, id_s0])
            elif st_0.peak < st_1.peak and abs(st_1.mean - st_0.mean) < st_1.sigma:
                tmp_list.append(id_s0)
                old_to_new_map.append([id_s0, id_s1 + id_s0 + 1])

    # Step 2: Remove states that don't meet the "final" criterion from the 'list_of_states'.
    # Note: The loop iterates in reverse to avoid index errors when removing elements.
    tmp_list = np.unique(tmp_list)[::-1]
    for state_id in tmp_list:
        list_of_states.pop(state_id)

    list_of_states = sorted(list_of_states, key=lambda x: x.mean)

    # Relabel accorind to the new states
    for mol_id in range(all_the_labels.shape[0]):
        for window_id in range(all_the_labels.shape[1]):
            for pair_id in range(len(old_to_new_map[::-1])):
                if all_the_labels[mol_id][window_id] == old_to_new_map[pair_id][0] + 1:
                    all_the_labels[mol_id][window_id] = old_to_new_map[pair_id][1] + 1

    list_unique = np.unique(all_the_labels)
    label_to_index = {label: index for index, label in enumerate(list_unique)}
    new_labels = np.vectorize(label_to_index.get)(all_the_labels)

    # Step 3: Calculate the final threshold values
    # and their types based on the intercept between neighboring states.
    list_of_states[0].th_inf[0] = m_range[0]
    list_of_states[0].th_inf[1] = 0

    for i in range(len(list_of_states) - 1):
        st_0 = list_of_states[i]
        st_1 = list_of_states[i + 1]
        coeff_a = st_1.sigma**2 - st_0.sigma**2
        coeff_b = -2*(st_0.mean*st_1.sigma**2 - st_1.mean*st_0.sigma**2)
        tmp_c = np.log(st_0.area*st_1.sigma/st_1.area/st_0.sigma)
        coeff_c = ((st_0.mean*st_1.sigma)**2
            - (st_1.mean*st_0.sigma)**2
            - ((st_0.sigma*st_1.sigma)**2)*tmp_c)
        delta = coeff_b**2 - 4*coeff_a*coeff_c
        # Determine the type of the threshold (0, 1 or 2).
        if coeff_a == 0.0:
            only_th = ((st_0.mean + st_1.mean)/2
                - st_0.sigma**2/2/(st_1.mean - st_0.mean)*np.log(st_0.area/st_1.area))
            list_of_states[i].th_sup[0] = only_th
            list_of_states[i].th_sup[1] = 1
            list_of_states[i + 1].th_inf[0] = only_th
            list_of_states[i + 1].th_inf[1] = 1
        elif delta >= 0:
            th_plus = (- coeff_b + np.sqrt(delta))/(2*coeff_a)
            th_minus = (- coeff_b - np.sqrt(delta))/(2*coeff_a)
            intercept_plus = gaussian(th_plus, st_0.mean, st_0.sigma, st_0.area)
            intercept_minus = gaussian(th_minus, st_0.mean, st_0.sigma, st_0.area)
            if intercept_plus >= intercept_minus:
                list_of_states[i].th_sup[0] = th_plus
                list_of_states[i].th_sup[1] = 1
                list_of_states[i + 1].th_inf[0] = th_plus
                list_of_states[i + 1].th_inf[1] = 1
            else:
                list_of_states[i].th_sup[0] = th_minus
                list_of_states[i].th_sup[1] = 1
                list_of_states[i + 1].th_inf[0] = th_minus
                list_of_states[i + 1].th_inf[1] = 1
        else:
            th_aver = (st_0.mean/st_0.sigma + st_1.mean/st_1.sigma)/(1/st_0.sigma + 1/st_1.sigma)
            list_of_states[i].th_sup[0] = th_aver
            list_of_states[i].th_sup[1] = 2
            list_of_states[i + 1].th_inf[0] = th_aver
            list_of_states[i + 1].th_inf[1] = 2

    list_of_states[-1].th_sup[0] = m_range[1]
    list_of_states[-1].th_sup[1] = 0

    # Step 4: Write the final states and final thresholds to text files.
    # The data is saved in two separate files: 'final_states.txt' and 'final_thresholds.txt'.
    with open('final_states.txt', 'w', encoding="utf-8") as file:
        print('# Mu \t Sigma \t A \t state_fraction', file=file)
        for state in list_of_states:
            print(state.mean, state.sigma, state.area, state.perc, file=file)
    with open('final_thresholds.txt', 'w', encoding="utf-8") as file:
        for state in list_of_states:
            print(state.th_inf[0], state.th_sup[0], file=file)

    # Step 5: Return the 'list_of_states' as the output of the function.
    return list_of_states, new_labels

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

    Steps:
    1. Sorts states based on relevance (StateMulti.perc).
    2. Relabels all the labels according to the new ordering.
    3. Merges strongly overlapping states based on mean and standard deviation comparisons.
    4. Removes gaps in the labeling and updates state percentages.
    5. Prints information about the final states to 'final_states.txt'.
    6. Returns the updated labels and modified list of StateMulti objects.
    """

    ### Step 1: sort according to the relevance
    sorted_indices = [index + 1 for index, _ in sorted(enumerate(states_list),
        key=lambda x: x[1].perc, reverse=True)]
    sorted_states = sorted(states_list, key=lambda x: x.perc, reverse=True)

    ### Step 2: relabel all the labels according to the new ordering
    sorted_all_the_labels = np.empty(all_the_labels.shape)
    for i, mol in enumerate(all_the_labels):
        for j, mol_t in enumerate(mol):
            for id0, new_id0 in enumerate(sorted_indices):
                if mol_t == new_id0:
                    sorted_all_the_labels[i][j] = id0 + 1
                    break
                sorted_all_the_labels[i][j] = 0

    ### Step 3: merge strongly overlapping states. Two states are merged if,
    ### along all the directions, their means dist less than the larger
    ### of their standard deviations in that direction.
    ### Find all the pairs of states which should be merged
    merge_pairs = []
    for i, st_0 in enumerate(sorted_states):
        for j, st_1 in enumerate(sorted_states[i + 1:]):
            diff = np.abs(np.subtract(st_1.mean, st_0.mean))
            if np.all(diff < [ max(st_0.sigma[k], st_1.sigma[k]) for k in range(diff.size) ]):
                merge_pairs.append([i + 1, j + i + 2])

    ## If a state can be merged with more than one state, choose the closest one ###
    pair_to_delete = []

    list_of_distances = []
    for pair in merge_pairs:
        st_0 = sorted_states[pair[0] - 1]
        st_1 = sorted_states[pair[1] - 1]
        diff = st_1.mean - st_0.mean
        dist = sum(pow(x, 2) for x in diff)
        list_of_distances.append(dist)

    for i, pair0 in enumerate(merge_pairs):
        list_of_possible_merging = []
        for j, pair1 in enumerate(merge_pairs):
            if pair1[1] == pair0[1]:
                list_of_possible_merging.append([j, list_of_distances[j]])

        tmp_array = np.array(list_of_possible_merging)
        best_state = np.argmin(tmp_array[:, 1])
        for i, elem in enumerate(list_of_possible_merging):
            if i != best_state:
                pair_to_delete.append(elem[0])

    for elem in np.unique(pair_to_delete)[::-1]:
        merge_pairs.pop(elem)

    for i, pair0 in enumerate(merge_pairs):
        for j, pair1 in enumerate(merge_pairs[i + 1:]):
            if pair1[0] == pair0[1]:
                merge_pairs[j][0] = pair0[0]

    ## Create a dictionary to easily relabel data points
    state_mapping = {i: i for i in range(len(sorted_states) + 1)}
    for id_s0, id_s1 in merge_pairs:
        state_mapping[id_s1] = id_s0

    ## Relabel the data points
    updated_labels = np.empty(sorted_all_the_labels.shape)
    for i, mol in enumerate(sorted_all_the_labels):
        for j, label in enumerate(mol):
            try:
                updated_labels[i][j] = state_mapping[label]
            except:
                continue

    ## Update the list of states
    states_to_remove = set(s1 for s0, s1 in merge_pairs)
    updated_states = [
        sorted_states[s] for s in range(len(sorted_states))
        if s + 1 not in states_to_remove
    ]

    ### Step 4: remove gaps in the labeling
    current_labels = np.unique(updated_labels)
    if current_labels[0] != 0:
        current_labels = np.insert(current_labels, 0, 0)
    for i, mol in enumerate(updated_labels):
        for j, label in enumerate(mol):
            for k, curr_lab in enumerate(current_labels):
                if label == curr_lab:
                    updated_labels[i][j] = k

    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(updated_labels == st_id + 1)
        state.perc = num_of_points / updated_labels.size

    ### Step 5: print informations on the final states
    with open('final_states.txt', 'w', encoding="utf-8") as file:
        print('#center_coords, semiaxis, fraction_of_data', file=file)
        for state in updated_states:
            center = state.mean
            centers = '[' + str(center[0]) + ', '
            for tmp in center[1:-1]:
                centers += str(tmp) + ', '
            centers += str(center[-1]) + ']'
            area = state.area
            axis = '[' + str(area[0]) + ', '
            for tmp in area[1:-1]:
                axis += str(tmp) + ', '
            axis += str(area[-1]) + ']'
            print(centers, axis, state.perc, file=file)

    return updated_labels, updated_states

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

def plot_tra_figure(number_of_states: np.ndarray, fraction_0: np.ndarray, par: Parameters, show_plot: bool):
    """
    Plots time resolution analysis figures based on the number of states
    and fraction of a specific state.

    Args:
    - number_of_states (np.ndarray): Array containing the number of states.
    - fraction_0 (np.ndarray): Array containing the fraction of a specific state.
    - par (Parameters): Instance of Parameters with time-related parameters.
    - show_plot (bool): Flag to display the plot if True.

    Steps:
    - Extracts time conversion and units from the provided Parameters instance.
    - Performs necessary data manipulations on the number of states and fractions.
    - Plots the analysis figure with customizable options for specific or averaged data views.
    - Saves the figure as 'Time_resolution_analysis.png' at high resolution.

    Note:
    - Provides options for specific or averaged data views.
    - Offers control over plot settings like axes, labels, and visualization options.
    - If 'show_plot' is True, the generated plot will be displayed.
    """

    t_conv, units = par.t_conv, par.t_units
    number_of_states = np.array(number_of_states)
    time = np.array(number_of_states.T[0])*t_conv
    number_of_states = number_of_states[:, 1:]
    fraction_0 = np.array(fraction_0)[:, 1:]

    fig, ax = plt.subplots()
    ### If I want to chose one particular value of the smoothing: #########
    t_smooth_idx = 0
    y_signal = number_of_states.T[t_smooth_idx]
    y_2 = fraction_0.T[t_smooth_idx]
    #######################################################################

    # ### If I want to average over the different smoothings: ###############
    # y_signal = np.mean(number_of_states, axis=1)
    # y_err = np.std(number_of_states, axis=1)
    # y_inf = y_signal - y_err
    # y_sup = y_signal + y_err
    # ax.fill_between(time, y_inf, y_sup, zorder=0, alpha=0.4, color='gray')
    # y_2 = np.mean(fraction_0, axis=1)
    # #######################################################################

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

    if show_plot:
        plt.show()
    fig.savefig('output_figures/Time_resolution_analysis.png', dpi=600)

def sankey(all_the_labels: np.ndarray, tmp_frame_list: list[int], par: Parameters, filename: str, show_plot: bool):
    """
    Computes and plots a Sankey diagram based on provided data and parameters.

    Args:
    - all_the_labels (np.ndarray): Array containing labels for each frame.
    - tmp_frame_list (list[int]): List of frame indices.
    - par (Parameters): Instance of Parameters containing time-related parameters.
    - filename (str): Name of the file for the output Sankey diagram.
    - show_plot (bool): Flag to display the plot if True.

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
    n_states = np.unique(all_the_labels).size

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
            trans_mat[int(label[t_0])][int(label[t_0 + t_jump])] += 1

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

    if show_plot:
        fig.show()
    fig.write_image('output_figures/' + filename + '.png', scale=5.0)

def plot_state_populations(all_the_labels: np.ndarray, par: Parameters, filename: str, show_plot: bool):
    """
    Plots the populations of states over time.

    Args:
    - all_the_labels (np.ndarray): Array containing state labels for each time step.
    - par (Parameters): Instance of Parameters class containing time-related information.
    - filename (str): Name of the file to save the generated plot.
    - show_plot (bool): Flag to display the plot if True.

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
    t_steps = all_the_labels.shape[1]
    time = par.print_time(t_steps)
    list_of_populations = []
    for label in np.unique(all_the_labels):
        population = []
        for i in range(t_steps):
            population.append(sum(all_the_labels[:, i] == label))
        list_of_populations.append(np.array(population)/num_part)

    # Generate the color palette.
    palette = []
    n_states = np.unique(all_the_labels).size
    cmap = plt.get_cmap('viridis', n_states)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    fig, ax = plt.subplots()
    for label, pop in enumerate(list_of_populations):
        ax.plot(time, pop, label='ENV' + str(label), color=palette[label])
    ax.set_xlabel(r'Time ' + par.t_units)
    ax.set_ylabel(r'ENV population')
    ax.legend()

    if show_plot:
        plt.show()
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
                        int(all_the_labels[i][j]), j + 1, file=file)
                else:
                    print(m_clean[i][j][0], m_clean[i][j][1], m_clean[i][j][2],
                        int(all_the_labels[i][j]), j + 1, file=file)

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
        with open(trj_file, "r", encoding="utf-8") as in_file:
            tmp = [ line.split() for line in in_file.readlines() ]

        num_of_particles = all_the_labels.shape[0]
        total_time = all_the_labels.shape[1]
        nlines = (num_of_particles + 2)*total_time

        print('\t Removing the first', int(par.t_smooth/2) + par.t_delay, 'frames...')
        for _ in range(int(par.t_smooth/2) + par.t_delay):
            for _ in range(num_of_particles + 2):
                tmp.pop(0)

        print('\t Removing the last', int((len(tmp) - nlines)/(num_of_particles + 2)), 'frames...')
        while len(tmp) > nlines:
            tmp.pop(-1)

        with open('colored_trj.xyz', "w+", encoding="utf-8") as out_file:
            i = 0
            for j in range(total_time):
                print(tmp[i][0], file=out_file)
                print(tmp[i + 1][0], file=out_file)
                for k in range(num_of_particles):
                    print(all_the_labels[k][j],
                        tmp[i + 2 + k][1], tmp[i + 2 + k][2], tmp[i + 2 + k][3], file=out_file)
                i += num_of_particles + 2
    else:
        print('No ' + trj_file + ' found for coloring the trajectory.')
