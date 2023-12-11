"""
Code for clustering of multivariate (2- or 3-dimensional) time-series data.
See the documentation for all the details.
"""
import copy as copy_name
import shutil
from typing import Union
from matplotlib.image import NonUniformImage
from matplotlib.patches import Ellipse
from functions import *

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = 'states_output.txt'
SHOW_PLOT = False

def all_the_input_stuff():
    """
    Perform input-related operations: reading input parameters, data, and creating output files.

    Returns:
    - tuple or None: Tuple containing:
        - list or None: Processed raw data from input directories, or None if an error occurs.
        - Parameters or None: Input parameters instance, or None if an error occurs.

    This function handles various input-related tasks:
    - Reads input parameters from 'input_parameters.txt' file.
    - Reads raw data from specified directories, removing initial frames based on 'tau_delay'.
    - Checks if the shape of raw data arrays matches.
    - Creates an output file and clears an existing 'output_figures' folder.

    It returns the processed raw data and input parameters for further analysis,
    or None if errors occur.
    """
    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters('input_parameters.txt')
    par.print_to_screen()

    tmp_m_raw = []
    for data_dir in data_directory:
        # Read raw data from the specified directory/files.
        m_raw = read_data(data_dir)

        # Remove initial frames based on 'tau_delay'.
        m_raw = m_raw[:, par.t_delay:]

        tmp_m_raw.append(m_raw)

    for dim in range(len(tmp_m_raw) - 1):
        if tmp_m_raw[dim].shape != tmp_m_raw[dim + 1].shape :
            print('ERROR: The signals do not correspond. Abort.')
            return None, None

    ### Create files for output
    with open(OUTPUT_FILE, 'w', encoding="utf-8") as file:
        file.write('#')
    figures_folder = 'output_figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    for filename in os.listdir(figures_folder):
        file_path = os.path.join(figures_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exc_msg:
            print(f'Failed to delete {file_path}. Reason: {exc_msg}')

    # Return required data for further analysis.
    return tmp_m_raw, par

def preparing_the_data(tmp_m_raw: np.ndarray, par: Parameters):
    """
    Prepare the raw data for analysis by applying filtering and normalization.

    Parameters:
    - tmp_m_raw (np.ndarray): Array containing raw data.
    - par (Parameters): Instance of Parameters class containing required parameters.

    Returns:
    - tuple: A tuple containing:
        - np.ndarray: Processed array of cleaned data for analysis.
        - list: List containing minimum and maximum values
        for each dimension of the processed data.

    This function prepares the raw data for analysis by performing the following steps:
    - Applies a moving average filter on the raw data.
    - Normalizes the data to the range [0, 1] (commented out in the code).
    - Transposes and rearranges the data for further processing.
    - Calculates the number of windows for analysis based on parameters.
    - Prints informative messages about trajectory details.

    Returns the processed data and limits (minimum and maximum values for each dimension).
    """
    tau_window, t_smooth, t_conv, t_units = par.tau_w, par.t_smooth, par.t_conv, par.t_units

    m_clean = []
    for m_raw in tmp_m_raw:
        # Apply filtering on the data
        tmp_m = moving_average(m_raw, t_smooth)

        # Normalize the data to the range [0, 1].
        # sig_max = np.max(tmp_m)
        # sig_min = np.min(tmp_m)
        # tmp_m = (tmp_m - sig_min)/(sig_max - sig_min)

        m_clean.append(tmp_m)

    m_arr = np.array(m_clean)
    m_limits = [ [np.min(x), np.max(x) ] for x in m_arr ]
    m_arr = np.transpose(m_arr, axes=(1, 2, 0))

    total_particles = m_arr.shape[0]
    total_time = m_arr.shape[1]
    # Calculate the number of windows for the analysis.
    num_windows = int(total_time / tau_window)

    # Print informative messages about trajectory details.
    print('\tTrajectory has ' + str(total_particles) + ' particles. ')
    print('\tTrajectory of length ' + str(total_time) + ' frames (' +
        str(total_time*t_conv) + ' ' + t_units + ').')
    print('\tUsing ' + str(num_windows) + ' windows of length ' +
        str(tau_window) + ' frames (' + str(tau_window*t_conv) + ' ' + t_units + ').')

    return m_arr, m_limits

def plot_input_data(m_clean: np.ndarray, par: Parameters, filename: str):
    """
    Plot input data: histograms and trajectories.

    Parameters:
    - m_clean (np.ndarray): Processed array of cleaned data for plotting.
    - par (Parameters): Instance of Parameters class containing required parameters.
    - filename (str): Name of the output file to save the plot.

    This function creates plots for input data:
    - For 2D data: Creates histograms and individual trajectories (side-by-side).
    - For 3D data: Creates a 3D plot showing the trajectories.

    The function uses the processed data and parameters to generate visualizations
    and saves the plot as an image file.
    """
    bin_selection = []
    counts_selection = []
    for dim in range(m_clean.shape[2]):
        # Flatten the m matrix and compute histogram counts and bins
        flat_m = m_clean[:,:,dim].flatten()
        counts0, bins0 = np.histogram(flat_m, bins=par.bins, density=True)
        counts0 *= flat_m.size
        bin_selection.append(bins0)
        counts_selection.append(counts0)

    if m_clean.shape[2] == 2:
        # Create a plot with two subplots (side-by-side)
        fig = plt.figure(figsize=(9, 9))
        grid = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(grid[0:1, 0:3])
        ax2 = fig.add_subplot(grid[1:4, 0:3])
        ax3 = fig.add_subplot(grid[1:4, 3:4])
        ax1.set_xticklabels([])
        ax3.set_yticklabels([])

        # Plot histograms
        ax1.stairs(counts_selection[0], bin_selection[0], fill=True)
        ax3.stairs(counts_selection[1], bin_selection[1], fill=True, orientation='horizontal')

        # Plot the individual trajectories in the first subplot (left side)
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m_clean):
            if np.max(mol) == np.max(m_clean):
                id_max = idx
            if np.min(mol) == np.min(m_clean):
                id_min = idx
        step = 10 if m_clean.size > 1000000 else 1
        for idx, mol in enumerate(m_clean[::step]):
            ax2.plot(mol[:,0], mol[:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)
        ax2.plot(m_clean[id_min][:,0], m_clean[id_min][:,1],
            color='black', lw=0.1, alpha=0.5, rasterized=True)
        ax2.plot(m_clean[id_max][:,0], m_clean[id_max][:,1],
            color='black', lw=0.1, alpha=0.5, rasterized=True)

        # Set labels and titles for the plots
        ax2.set_ylabel('Signal 1')
        ax2.set_xlabel('Signal 2')

    elif m_clean.shape[2] == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')

        # Plot the individual trajectories
        step = 1 if m_clean.size > 1000000 else 1
        for idx, mol in enumerate(m_clean[::step]):
            ax.plot(mol[:,0], mol[:,1], mol[:,2], color='black', marker='o',
                ms=0.5, lw=0.2, alpha=1.0, rasterized=True)

        # Set labels and titles for the plots
        ax.set_xlabel('Signal 1')
        ax.set_ylabel('Signal 2')
        ax.set_zlabel('Signal 3')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def gauss_fit_max(m_clean: np.ndarray, m_limits: list[list[int]], bins: Union[int, str], filename: str):
    """
    Perform Gaussian fit and generate plots based on the provided data.

    Parameters:
    - m_clean (np.ndarray): Processed array of cleaned data.
    - m_limits (list[list[int]]): List containing minimum and maximum values for each dimension.
    - bins (Union[int, str]): Number of bins for histograms or 'auto' for automatic binning.
    - filename (str): Name of the output file to save the plot.

    Returns:
    - State or None: State object for state identification or None if fitting fails.

    This function performs the following steps:
    1. Generates histograms based on the data and chosen binning strategy.
    2. Smoothes the histograms.
    3. Identifies the maximum values in the histograms.
    4. Finds minima surrounding the maximum values.
    5. Tries fitting between minima and checks goodness.
    6. Determines the interval of half height.
    7. Tries fitting between the half-height interval and checks goodness.
    8. Chooses the best fit based on goodness and provides state identification.

    The function then generates plots of the distribution and fitted Gaussians
    based on the dimensionality of the data. It saves the plot as an image file
    and returns a State object for further analysis or None if fitting fails.
    """
    print('* Gaussian fit...')
    flat_m = m_clean.reshape((m_clean.shape[0]*m_clean.shape[1], m_clean.shape[2]))

    ### 1. Histogram with 'auto' binning ###
    if bins == 'auto':
        bins = max(int(np.power(m_clean.size, 1/3)*2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)
    gap = 1
    if np.all([e.size > 40 for e in edges]):
        gap = 3

    ### 2. Smoothing with tau = 3 ###
    counts = moving_average_2d(counts, gap)

    ### 3. Find the maximum ###
    def find_max_index(data: np.ndarray):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    # max_val = counts.max()
    max_ind = find_max_index(counts)

    ### 4. Find the minima surrounding it ###
    def find_minima_around_max(data: np.ndarray, max_ind: tuple, gap: int):
        """
        Find minima surrounding the maximum value in the given data array.

        Parameters:
        - data (np.ndarray): Input data array.
        - max_ind (tuple): Indices of the maximum value in the data.
        - gap (int): Gap value to determine the search range around the maximum.

        Returns:
        - list: List of indices representing the minima surrounding the maximum in each dimension.

        This function finds minima around the maximum value in the given data array
        for each dimension.
        The function returns a list containing the indices of minima in each dimension.
        """
        minima = []

        for dim in range(data.ndim):
            min_id0 = max(max_ind[dim] - gap, 0)
            min_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

            tmp_max1 = copy_name.deepcopy(max_ind)
            tmp_max2 = copy_name.deepcopy(max_ind)

            tmp_max1[dim] = min_id0
            tmp_max2[dim] = min_id0 - 1
            while min_id0 > 0 and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]:
                tmp_max1[dim] -= 1
                tmp_max2[dim] -= 1
                min_id0 -= 1

            tmp_max1 = copy_name.deepcopy(max_ind)
            tmp_max2 = copy_name.deepcopy(max_ind)

            tmp_max1[dim] = min_id1
            tmp_max2[dim] = min_id1 + 1
            while min_id1 < data.shape[dim] - 1 and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]:
                tmp_max1[dim] += 1
                tmp_max2[dim] += 1
                min_id1 += 1

            minima.extend([min_id0, min_id1])

        return minima

    minima = find_minima_around_max(counts, max_ind, gap)

    ### 5. Try the fit between the minima and check its goodness ###
    popt_min = []
    goodness_min = 0
    for dim in range(m_clean.shape[2]):
        try:
            flag_min, goodness, popt = custom_fit(dim, max_ind[dim], minima,
                edges[dim], counts, gap, m_limits)
            popt[2] *= flat_m.T[0].size
            popt_min.extend(popt)
            goodness_min += goodness
        except:
            popt_min = []
            flag_min = False
            goodness_min -= 5

    ### 6. Find the interval of half height ###
    def find_half_height_around_max(data, max_ind, gap):
        """
        Find indices around the maximum value where the data reaches half of its maximum value.

        Parameters:
        - data (np.ndarray): Input data array.
        - max_ind (tuple): Indices of the maximum value in the data.
        - gap (int): Gap value to determine the search range around the maximum.

        Returns:
        - list: List of indices representing the range around the maximum where data
        reaches half of its maximum value.

        This function identifies the indices around the maximum value in the given data array
        where the data reaches half of its maximum value. The function returns a list
        containing the indices representing the range where the data reaches half of its
        maximum value in each dimension.
        """
        max_val = data.max()
        minima = []

        for dim in range(data.ndim):
            half_id0 = max(max_ind[dim] - gap, 0)
            half_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

            tmp_max = copy_name.deepcopy(max_ind)

            tmp_max[dim] = half_id0
            while half_id0 > 0 and data[tuple(tmp_max)] > max_val/2:
                tmp_max[dim] -= 1
                half_id0 -= 1

            tmp_max = copy_name.deepcopy(max_ind)

            tmp_max[dim] = half_id1
            while half_id1 < data.shape[dim] - 1 and data[tuple(tmp_max)] > max_val/2:
                tmp_max[dim] += 1
                half_id1 += 1

            minima.extend([half_id0, half_id1])

        return minima

    minima = find_half_height_around_max(counts, max_ind, gap)

    ### 7. Try the fit between the minima and check its goodness ###
    popt_half = []
    goodness_half = 0
    for dim in range(m_clean.shape[2]):
        try:
            flag_half, goodness, popt = custom_fit(dim, max_ind[dim], minima,
                edges[dim], counts, gap, m_limits)
            popt[2] *= flat_m.T[0].size
            popt_half.extend(popt)
            goodness_half += goodness
        except:
            popt_half = []
            flag_half = False
            goodness_half -= 5

    ### 8. Choose the best fit ###
    goodness = goodness_min
    if flag_min == 1 and flag_half == 0:
        popt = popt_min
    elif flag_min == 0 and flag_half == 1:
        popt = popt_half
        goodness = goodness_half
    elif flag_min*flag_half == 1:
        if goodness_min >= goodness_half:
            popt = popt_min
        else:
            popt = popt_half
            goodness = goodness_half
    else:
        print('\tWARNING: this fit is not converging.')
        return None

    if len(popt) != m_clean.shape[2]*3:
        print('\tWARNING: this fit is not converging.')
        return None

    ### Find the tresholds for state identification
    mean, sigma, area = [], [], []
    for dim in range(m_clean.shape[2]):
        mean.append(popt[3*dim])
        sigma.append(popt[3*dim + 1])
        area.append(popt[3*dim + 2])
    state = StateMulti(np.array(mean), np.array(sigma), np.array(area))
    state.build_boundaries(NUMBER_OF_SIGMAS)

    ### Plot the distribution and the fitted Gaussians
    if m_clean.shape[2] == 2:
        with open(OUTPUT_FILE, 'a', encoding="utf-8") as file:
            print('\n', file=file)
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}')
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}', file=file)
            print('\tFit goodness = ' + str(goodness), file=file)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = NonUniformImage(ax, interpolation='nearest')
        xcenters = (edges[0][:-1] + edges[0][1:]) / 2
        ycenters = (edges[1][:-1] + edges[1][1:]) / 2
        im.set_data(xcenters, ycenters, counts.T)
        ax.add_image(im)
        ax.scatter(mean[0], mean[1], s=8.0, c='red')
        circle1 = Ellipse(tuple(mean), sigma[0], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple(mean), state.axis[0], state.axis[1], color='r', fill=False)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.set_xlim(m_limits[0][0], m_limits[0][1])
        ax.set_ylim(m_limits[1][0], m_limits[1][1])
    elif m_clean.shape[2] == 3:
        with open(OUTPUT_FILE, 'a', encoding="utf-8") as file:
            print('\n', file=file)
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], '
                f'sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], '
                f'area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}')
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], '
                f'sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], '
                f'area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}', file=file)
            print('\tFit goodness = ' + str(goodness), file=file)

        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        xcenters = (edges[0][:-1] + edges[0][1:]) / 2
        ycenters = (edges[1][:-1] + edges[1][1:]) / 2
        zcenters = (edges[2][:-1] + edges[2][1:]) / 2

        im = NonUniformImage(ax[0][0], interpolation='nearest')
        im.set_data(xcenters, ycenters, np.sum(counts, axis=0))
        ax[0][0].add_image(im)
        ax[0][0].scatter(mean[0], mean[1], s=8.0, c='red')
        circle1 = Ellipse(tuple([mean[0], mean[1]]), sigma[0], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple([mean[0], mean[1]]), state.axis[0], state.axis[1],
            color='r', fill=False)
        ax[0][0].add_patch(circle1)
        ax[0][0].add_patch(circle2)

        im = NonUniformImage(ax[0][1], interpolation='nearest')
        im.set_data(zcenters, ycenters, np.sum(counts, axis=1))
        ax[0][1].add_image(im)
        ax[0][1].scatter(mean[2], mean[1], s=8.0, c='red')
        circle1 = Ellipse(tuple([mean[2], mean[1]]), sigma[2], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple([mean[2], mean[1]]), state.axis[2], state.axis[1],
            color='r', fill=False)
        ax[0][1].add_patch(circle1)
        ax[0][1].add_patch(circle2)

        im = NonUniformImage(ax[1][0], interpolation='nearest')
        im.set_data(xcenters, zcenters, np.sum(counts, axis=2))
        ax[1][0].add_image(im)
        ax[1][0].scatter(mean[0], mean[2], s=8.0, c='red')
        circle1 = Ellipse(tuple([mean[0], mean[2]]), sigma[0], sigma[2], color='r', fill=False)
        circle2 = Ellipse(tuple([mean[0], mean[2]]), state.axis[0], state.axis[2],
            color='r', fill=False)
        ax[1][0].add_patch(circle1)
        ax[1][0].add_patch(circle2)

        ax[0][0].set_xlim(m_limits[0][0], m_limits[0][1])
        ax[0][0].set_ylim(m_limits[1][0], m_limits[1][1])
        ax[0][1].set_xlim(m_limits[2][0], m_limits[2][1])
        ax[0][1].set_ylim(m_limits[1][0], m_limits[1][1])
        ax[1][0].set_xlim(m_limits[0][0], m_limits[0][1])
        ax[1][0].set_ylim(m_limits[2][0], m_limits[2][1])

    if SHOW_PLOT:
        plt.show()
    fig.savefig(filename + '.png', dpi=600)
    plt.close(fig)

    return state

def find_stable_trj(m_clean: np.ndarray, tau_window: int, state: StateMulti, all_the_labels: np.ndarray, offset: int):
    """
    Find stable windows in the trajectory based on provided data and state info.

    Parameters:
    - m_clean (np.ndarray): Cleaned trajectory data.
    - tau_window (int): Length of the window for analysis.
    - state (StateMulti): State information.
    - all_the_labels (np.ndarray): All labels for the trajectory.
    - offset (int): Offset for labeling stable windows.

    Returns:
    - np.ndarray: Array of non-stable windows.
    - float: Fraction of stable windows.
    - bool: Indicates one last state after finding stable windows.
    """
    print('* Finding stable windows...')

    # Calculate the number of windows in the trajectory
    number_of_windows = int(m_clean.shape[1] / tau_window )

    mask_unclassified = all_the_labels < 0.5
    m_reshaped = m_clean[:, :number_of_windows*tau_window].reshape(m_clean.shape[0],
        number_of_windows, tau_window, m_clean.shape[2])
    shifted = m_reshaped - state.mean
    rescaled = shifted / state.axis
    squared_distances = np.sum(rescaled**2, axis=3)
    mask_dist = np.max(squared_distances, axis=2) <= 1.0
    mask = mask_unclassified & mask_dist

    all_the_labels[mask] = offset + 1   # Label the stable windows in the new state
    counter = np.sum(mask)              # The number of stable windows found

    # Store non-stable windows in a list, for the next iteration
    m_new = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window*tau_window:(window + 1)*tau_window]
        m_new.append(r_w)

    # Calculate the fraction of stable windows found
    fraction_of_points = counter/(all_the_labels.size)

    # Print the fraction of stable windows
    with open(OUTPUT_FILE, 'a', encoding="utf-8") as file:
        print(f'\tFraction of windows in state {offset + 1} = {fraction_of_points:.3}')
        print(f'\tFraction of windows in state {offset + 1} = {fraction_of_points:.3}', file=file)

    # Convert the list of non-stable windows to a NumPy array
    m_new_arr = np.array(m_new)
    one_last_state = True
    if len(m_new_arr) == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows,
    # and the updated list_of_states
    return m_new_arr, fraction_of_points, one_last_state

def iterative_search(m_clean: np.ndarray, m_limits: list[list[int]], par: Parameters, name: str):
    """
    Perform an iterative search to identify stable windows in trajectory data.

    Args:
    - m_clean (np.ndarray): Cleaned trajectory data.
    - m_limits (list[list[int]]): Limits of the trajectory data.
    - par (Parameters): Parameters object containing tau_w and bins.
    - name (str): Name for the output figures.

    Returns:
    - all_the_labels (np.ndarray): Array of labels for each window.
    - list_of_states: List of identified states.
    - one_last_state (bool): Flag indicating the presence of one last state.
    """
    tau_w, bins = par.tau_w, par.bins

    # Initialize an array to store labels for each window.
    num_windows = int(m_clean.shape[1] / tau_w)
    all_the_labels = np.zeros((m_clean.shape[0], num_windows)).astype(int)

    states_list = []
    m_copy = m_clean
    iteration_id = 1
    states_counter = 0
    one_last_state = False
    while True:
        ### Locate and fit maximum in the signal distribution
        state = gauss_fit_max(m_copy, m_limits, bins, 'output_figures/' +
            name + 'Fig1_' + str(iteration_id))
        if state is None:
            print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')
            break

        ### Find the windows in which the trajectories are stable in the maximum
        m_new, counter, one_last_state = find_stable_trj(
            m_clean, tau_w, state, all_the_labels, states_counter)
        state.perc = counter

        if counter > 0.0:
            states_list.append(state)

        states_counter += 1
        iteration_id += 1
        ### Exit the loop if no new stable windows are found
        if counter <= 0.0:
            print('Iterations interrupted because no data points assigned to last state. ')
            break
        if m_new.size == 0:
            print('Iterations interrupted because all data points assigned to one state. ')
            break
        m_copy = m_new

    all_the_labels, list_of_states = relabel_states_2d(all_the_labels, states_list)
    return all_the_labels, list_of_states, one_last_state

def plot_cumulative_figure(m_clean: np.ndarray, all_the_labels: np.ndarray, list_of_states: list[StateMulti], filename: str):
    """
    Plot a cumulative figure displaying trajectories and identified states.

    Args:
    - m_clean (np.ndarray): Cleaned trajectory data.
    - all_the_labels (np.ndarray): Array of labels for each window.
    - list_of_states (list[StateMulti]): List of identified states.
    - filename (str): Name for the output figure.

    Returns:
    - None
    """
    print('* Printing cumulative figure...')
    colormap = 'viridis'
    n_states = len(list_of_states) + 1
    tmp = plt.get_cmap(colormap, n_states)
    colors_from_cmap = tmp(np.arange(0, 1, 1/n_states))
    colors_from_cmap[-1] = tmp(1.0)

    fig = plt.figure(figsize=(6, 6))
    if m_clean.shape[2] == 3:
        ax = plt.axes(projection='3d')

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m_clean):
            if np.max(mol) == np.max(m_clean):
                id_max = idx
            if np.min(mol) == np.min(m_clean):
                id_min = idx

        lw = 0.05
        max_t = all_the_labels.shape[1]
        m_resized = m_clean[:, :max_t:, :]

        step = 5 if m_resized.size > 1000000 else 1
        for i, mol in enumerate(m_resized[::step]):
            ax.plot(mol.T[0], mol.T[1], mol.T[2], c='black', lw=lw, rasterized=True, zorder=0)
            color_list = all_the_labels[i*step]
            ax.scatter(mol.T[0], mol.T[1], mol.T[2],
                c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)

        color_list = all_the_labels[id_min]
        ax.plot(m_resized[id_min].T[0], m_resized[id_min].T[1],
            m_resized[id_min].T[2], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m_resized[id_min].T[0], m_resized[id_min].T[1], m_resized[id_min].T[2],
            c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)
        color_list = all_the_labels[id_max]
        ax.plot(m_resized[id_max].T[0], m_resized[id_max].T[1],
            m_resized[id_max].T[2], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m_resized[id_max].T[0], m_resized[id_max].T[1], m_resized[id_max].T[2],
            c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)

        # Plot the Gaussian distributions of states
        for s_id, state in enumerate(list_of_states):
            ang_u = np.linspace(0, 2*np.pi, 100)
            ang_v = np.linspace(0, np.pi, 100)
            point_x = state.axis[0]*np.outer(np.cos(ang_u), np.sin(ang_v)) + state.mean[0]
            point_y = state.axis[1]*np.outer(np.sin(ang_u), np.sin(ang_v)) + state.mean[1]
            point_z = state.axis[2]*np.outer(np.ones_like(ang_u), np.cos(ang_v)) + state.mean[2]
            ax.plot_surface(point_x, point_y, point_z, alpha=0.25, color=colors_from_cmap[s_id + 1])

        # Set plot titles and axis labels
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
    elif m_clean.shape[2] == 2:
        ax = plt.axes()

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m_clean):
            if np.max(mol) == np.max(m_clean):
                id_max = idx
            if np.min(mol) == np.min(m_clean):
                id_min = idx

        lw = 0.05
        max_t = all_the_labels.shape[1]
        m_resized = m_clean[:, :max_t:, :]
        step = 5 if m_resized.size > 1000000 else 1

        for i, mol in enumerate(m_resized[::step]):
            ax.plot(mol.T[0], mol.T[1], c='black', lw=lw, rasterized=True, zorder=0)
            color_list = all_the_labels[i*step]
            ax.scatter(mol.T[0], mol.T[1], c=color_list, cmap=colormap, vmin=0, vmax=n_states-1,
                s=0.5, rasterized=True)

        color_list = all_the_labels[id_min]
        ax.plot(m_resized[id_min].T[0], m_resized[id_min].T[1],
            c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m_resized[id_min].T[0], m_resized[id_min].T[1],
            c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
        color_list = all_the_labels[id_max]
        ax.plot(m_resized[id_max].T[0], m_resized[id_max].T[1],
            c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m_resized[id_max].T[0], m_resized[id_max].T[1],
            c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

        # Plot the Gaussian distributions of states
        for state in list_of_states:
            ellipse = Ellipse(tuple(state.mean),
                state.axis[0], state.axis[1], color='black', fill=False)
            ax.add_patch(ellipse)

        # Set plot titles and axis labels
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def plot_one_trajectory(m_clean: np.ndarray, par: Parameters, all_the_labels: np.ndarray, filename: str):
    """
    Plot a single trajectory of an example particle with associated labels.

    Args:
    - m_clean (np.ndarray): Cleaned trajectory data.
    - par (Parameters): Parameters object containing example particle ID.
    - all_the_labels (np.ndarray): Array of labels for each window.
    - filename (str): Name for the output figure.

    Returns:
    - None
    """
    colormap = 'viridis'

    # Get the signal of the example particle
    signal_x = m_clean[par.example_id].T[0][:all_the_labels.shape[1]]
    signal_y = m_clean[par.example_id].T[1][:all_the_labels.shape[1]]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a colormap to map colors to the labels of the example particle
    cmap = plt.get_cmap(colormap,
        int(np.max(np.unique(all_the_labels)) - np.min(np.unique(all_the_labels)) + 1))
    color = all_the_labels[par.example_id]
    ax.plot(signal_x, signal_y, c='black', lw=0.1)

    ax.scatter(signal_x, signal_y, c=color, cmap=cmap, vmin=np.min(np.unique(all_the_labels)),
        vmax=np.max(np.unique(all_the_labels)), s=1.0, zorder=10)

    # Set plot titles and axis labels
    fig.suptitle('Example particle: ID = ' + str(par.example_id))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def timeseries_analysis(m_raw: np.ndarray, par: Parameters, tau_w: int, t_smooth: int):
    """
    Perform time series analysis on the input data.

    Args:
    - m_raw (np.ndarray): Raw input time series data.
    - par (Parameters): Parameters object containing necessary parameters.
    - tau_w (int): the time window for the analysis
    - t_smooth (int): the width of the moving average for the analysis

    Returns:
    - Tuple (int, float): Number of identified states, fraction of unclassified data.
    """

    print('* New analysis: ', tau_w, t_smooth)
    name = str(t_smooth) + '_' + str(tau_w) + '_'

    tmp_par = par.create_copy()
    tmp_par.tau_w = tau_w
    tmp_par.t_smooth = t_smooth

    m_clean, m_limits = preparing_the_data(m_raw, tmp_par)
    plot_input_data(m_clean, tmp_par, name + 'Fig0')

    all_the_labels, list_of_states, one_last_state = iterative_search(m_clean, m_limits, tmp_par, name)
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        # We need to free the memory otherwise it accumulates
        del m_raw
        del m_clean
        del all_the_labels
        return 1, 1.0

    # We need to free the memory otherwise it accumulates
    del m_raw
    del m_clean
    del all_the_labels

    fraction_0 = 1 - np.sum([ state.perc for state in list_of_states ])
    if one_last_state:
        print('Number of states identified:', len(list_of_states) + 1,
            '[' + str(fraction_0) + ']\n')
        return len(list_of_states) + 1, fraction_0

    print('Number of states identified:', len(list_of_states), '[' + str(fraction_0) + ']\n')
    return len(list_of_states), fraction_0

def compute_cluster_mean_seq(m_clean: np.ndarray, all_the_labels: np.ndarray, tau_window: int):
    """
    Plot the mean time sequence for clusters in the data.

    Args:
    - m_clean (np.ndarray): Cleaned input data.
    - all_the_labels (np.ndarray): Labels for each data point.
    - tau_window (int): Window size for time sequence.

    Returns:
    - None: If the third dimension of input data is greater than 2.
    """
    if m_clean.shape[2] > 2:
        return

    # Initialize lists to store cluster means and standard deviations
    center_list = []

    # Loop through unique labels (clusters)
    for ref_label in np.unique(all_the_labels):
        tmp = []
        # Iterate through molecules and their labels
        for i, mol in enumerate(all_the_labels):
            for j, label in enumerate(mol):
                # Define time interval
                t_0 = j*tau_window
                t_1 = (j + 1)*tau_window
                # If the label matches the current cluster, append the corresponding data to tmp
                if label == ref_label:
                    tmp.append(m_clean[i][t_0:t_1])

        # Calculate mean and standard deviation for the current cluster
        center_list.append(np.mean(tmp, axis=0))

    # Create a color palette
    palette = []
    cmap = plt.get_cmap('viridis', np.unique(all_the_labels).size)
    palette.append(rgb2hex(cmap(0)))
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Plot
    fig, ax = plt.subplots()
    for id_c, center in enumerate(center_list):
        sig_x = center[:, 0]
        sig_y = center[:, 1]
        ax.plot(sig_x, sig_y, label='ENV'+str(id_c), marker='o', c=palette[id_c])
    fig.suptitle('Average time sequence inside each environments')
    ax.set_xlabel(r'Signal 1')
    ax.set_ylabel(r'Signal 2')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/Fig4.png', dpi=600)

def full_output_analysis(m_raw: np.ndarray, par: Parameters):
    """
    Perform a comprehensive analysis and visualization pipeline on the input data.

    Args:
    - m_raw (np.ndarray): Raw input data.
    - par (Parameters): Parameters for analysis.

    Returns:
    - None
    """
    tau_w = par.tau_w
    m_clean, m_limits = preparing_the_data(m_raw, par)
    plot_input_data(m_clean, par, 'Fig0')

    all_the_labels, list_of_states, _ = iterative_search(m_clean, m_limits, par, '')
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        return

    compute_cluster_mean_seq(m_clean, all_the_labels, tau_w)
    plot_state_populations(all_the_labels, par, 'Fig5', SHOW_PLOT)
    # sankey(all_the_labels, [0, 100, 200, 300], par, 'Fig6', SHOW_PLOT)

    all_the_labels = assign_single_frames(all_the_labels, tau_w)

    plot_cumulative_figure(m_clean, all_the_labels, list_of_states, 'Fig2')
    plot_one_trajectory(m_clean, par, all_the_labels, 'Fig3')

    print_signal_with_labels(m_clean, all_the_labels)
    if os.path.exists('trajectory.xyz'):
        print_colored_trj_from_xyz('trajectory.xyz', all_the_labels, par)
    else:
        print_mol_labels_fbf_xyz(all_the_labels)

def time_resolution_analysis(m_raw: np.ndarray, par: Parameters, perform_anew: bool):
    """
    Analyze time series data with varying time resolution parameters.

    Args:
    - m_raw (np.ndarray): Raw time series data.
    - par (Parameters): Parameters for analysis.
    - perform_anew (bool): Flag to perform analysis anew or use saved results.

    Returns:
    - None
    """
    tau_window_list, t_smooth_list = param_grid(par, m_raw[0].shape[1])

    if perform_anew:
        ### If the analysis hat to be performed anew ###
        number_of_states = []
        fraction_0 = []
        for tau_w in tau_window_list:
            tmp = [tau_w]
            tmp1 = [tau_w]
            for t_s in t_smooth_list:
                n_s, f_0 = timeseries_analysis(m_raw, par, tau_w, t_s)
                tmp.append(n_s)
                tmp1.append(f_0)
            number_of_states.append(tmp)
            fraction_0.append(tmp1)
        number_of_states_arr = np.array(number_of_states)
        fraction_0_arr = np.array(fraction_0)

        np.savetxt('number_of_states.txt', number_of_states, fmt='%i',
            delimiter='\t', header='tau_window\n number_of_states for different t_smooth')
        np.savetxt('fraction_0.txt', fraction_0, delimiter=' ',
            header='tau_window\n fraction in ENV0 for different t_smooth')
    else:
        ### Otherwise, just do this ###
        number_of_states_arr = np.loadtxt('number_of_states.txt')
        fraction_0_arr = np.loadtxt('fraction_0.txt')

    plot_tra_figure(number_of_states_arr, fraction_0_arr, par, SHOW_PLOT)

def main():
    """
    all_the_input_stuff() reads the data and the parameters
    time_resolution_analysis() explore the parameter (tau_window, t_smooth) space.
        Use 'False' to skip it.
    full_output_analysis() performs a detailed analysis with the chosen parameters.
    """
    m_raw, par = all_the_input_stuff()
    time_resolution_analysis(m_raw, par, True)
    full_output_analysis(m_raw, par)

if __name__ == "__main__":
    main()
