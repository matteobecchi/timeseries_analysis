"""
Code for clustering of univariate time-series data. See the documentation for all the details.
"""
import shutil
from onion_clustering.functions import *

NUMBER_OF_SIGMAS = 2.0
OUTPUT_FILE = 'states_output.txt'

def all_the_input_stuff():
    """
    Reads input parameters and raw data from specified files and directories,
    processes the raw data, and creates output files.

    Returns:
    - data: Processed raw data after removing initial frames based on 't_delay'.
    - par: Object containing input parameters.

    Notes:
    - Ensure 'input_parameters.txt' exists and contains necessary parameters.
    - 'OUTPUT_FILE' constant specifies the output file.
    - 'tau_delay' parameter from 'input_parameters.txt' determines frames removal.
    - Creates 'output_figures' directory for storing output files.
    """

    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters('input_parameters.txt')
    par.print_to_screen()

    # Read raw data from the specified directory/files.
    if isinstance(data_directory, str):
        data = UniData(data_directory)
    else:
        print('\tERROR: data_directory.txt is missing or wrongly formatted. ')

    # Remove initial frames based on 't_delay'.
    data.remove_delay(par.t_delay)

    ### Create files for output
    with open(OUTPUT_FILE, 'w', encoding="utf-8") as file:
        print('#', file=file)
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
        except Exception as ex_msg:
            print(f'Failed to delete {file_path}. Reason: {ex_msg}')

    return data, par

def preparing_the_data(data: UniData, par: Parameters):
    """
    Processes raw data for analysis.

    Args:
    - data (UniData): Raw input data.
    - par (Parameters): Object containing parameters for data processing.

    Returns:
    - data (UniData): Cleaned and updated input data.
    """
    tau_window, t_smooth, t_conv, t_units = par.tau_w, par.t_smooth, par.t_conv, par.t_units

    # Apply filtering on the data
    data.smooth_mov_av(t_smooth)  # Smoothing using moving average
    # data.smooth_lpf(1/t_conv, t_smooth) # Smoothing using low-passing filter

    # Normalize the data to the range [0, 1]. Usually not needed. ###
    # data.normalize()

    # Calculate the number of windows for the analysis.
    num_windows = int(data.num_of_steps / tau_window)

    # Print informative messages about trajectory details.
    print('\tTrajectory has ' + str(data.num_of_particles) + ' particles. ')
    print('\tTrajectory of length ' + str(data.num_of_steps) +
        ' frames (' + str(data.num_of_steps*t_conv), t_units + ')')
    print('\tUsing ' + str(num_windows) + ' windows of length ' + str(tau_window) +
        ' frames (' + str(tau_window*t_conv), t_units + ')')

    return data

def plot_input_data(data: UniData, par: Parameters, filename: str):
    """
    Plots input data for visualization.

    Args:
    - data (UniData): Processed data for plotting.
    - par (Parameters): Object containing parameters for plotting.
    - filename (str): Name of the output plot file.
    """
    # Flatten the m_clean matrix and compute histogram counts and bins
    m_clean = data.matrix
    flat_m = m_clean.flatten()
    bins = par.bins
    counts, bins = np.histogram(flat_m, bins=bins, density=True)
    counts *= flat_m.size

    # Create a plot with two subplots (side-by-side)
    fig, ax = plt.subplots(1, 2, sharey=True,
        gridspec_kw={'width_ratios': [3, 1]},figsize=(9, 4.8))

    # Plot histogram in the second subplot (right side)
    ax[1].stairs(counts, bins, fill=True, orientation='horizontal')

    # Plot the individual trajectories in the first subplot (left side)
    time = par.print_time(m_clean.shape[1])
    step = 10 if m_clean.size > 1000000 else 1
    for mol in m_clean[::step]:
        ax[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5, rasterized=True)

    # Set labels and titles for the plots
    ax[0].set_ylabel('Signal')
    ax[0].set_xlabel(r'Simulation time $t$ ' + par.t_units)
    ax[1].set_xticklabels([])

    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def perform_gaussian_fit(
        id0: int, id1: int, max_ind: int, bins: np.ndarray,
        counts: np.ndarray, n_data: int, gap: int, interval_type: str
    ):
    """
    Perform Gaussian fit on given data within the specified range and parameters.

    Parameters:
    - id0 (int): Index representing the lower limit for data selection.
    - id1 (int): Index representing the upper limit for data selection.
    - bins (np.ndarray): Array containing bin values.
    - counts (np.ndarray): Array containing counts corresponding to bins.
    - n_data (int): Number of data points.
    - gap (int): Gap value for the fit.
    - interval_type (str): Type of interval.

    Returns:
    - tuple: A tuple containing:
        - bool: True if the fit is successful, False otherwise.
        - int: Goodness value calculated based on fit quality.
        - array or None: Parameters of the Gaussian fit if successful, None otherwise.

    The function performs a Gaussian fit on the specified data within the provided range.
    It assesses the goodness of the fit based on various criteria and returns the result.
    """
    goodness = 5
    selected_bins = bins[id0:id1]
    selected_counts = counts[id0:id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[id0] - bins[id1])/6
    area0 = counts[max_ind]*np.sqrt(np.pi)*sigma0
    try:
        popt, pcov = scipy.optimize.curve_fit(gaussian, selected_bins, selected_counts,
            p0=[mu0, sigma0, area0])
        if popt[1] < 0:
            popt[1] = -popt[1]
            popt[2] = -popt[2]
        gauss_max = popt[2]*np.sqrt(np.pi)*popt[1]
        if gauss_max < area0/2:
            goodness -= 1
        popt[2] *= n_data
        if popt[0] < selected_bins[0] or popt[0] > selected_bins[-1]:
            goodness -= 1
        if popt[1] > selected_bins[-1] - selected_bins[0]:
            goodness -= 1
        perr = np.sqrt(np.diag(pcov))
        for j, par_err in enumerate(perr):
            if par_err/popt[j] > 0.5:
                goodness -= 1
        if id1 - id0 <= gap:
            goodness -= 1
        return True, goodness, popt
    except RuntimeError:
        print('\t' + interval_type + ' fit: Runtime error. ')
        return False, goodness, None
    except TypeError:
        print('\t' + interval_type + ' fit: TypeError.')
        return False, goodness, None
    except ValueError:
        print('\t' + interval_type + ' fit: ValueError.')
        return False, goodness, None

def gauss_fit_max(m_clean: np.ndarray, par: Parameters, filename: str):
    """
    Performs Gaussian fitting on input data.

    Args:
    - m_clean (np.ndarray): Input data for Gaussian fitting.
    - par (Parameters): Object containing parameters for fitting.
    - filename (str): Name of the output plot file.

    Returns:
    - state (StateUni): Object containing Gaussian fit parameters (mu, sigma, area).

    Notes:
    - Requires 'bins' parameter in the 'par' object.
    - Performs Gaussian fitting on flattened input data.
    - Tries to find the maximum and fit Gaussians based on surrounding minima.
    - Chooses the best fit among the options or returns None if fitting fails.
    - Prints fit details and goodness of fit to an output file.
    - Generates a plot showing the distribution and the fitted Gaussian.
    """
    print('* Gaussian fit...')
    ######################################################
    # This is under development. Do not use it.
    # tmp_m = dense_interpolation(m_clean, dense_factor=2)
    # flat_m = tmp_m.flatten()
    ######################################################
    flat_m = m_clean.flatten()

    ### 1. Histogram ###
    counts, bins = np.histogram(flat_m, bins=par.bins, density=True)
    gap = 1
    if bins.size > 50:
        gap = 3

    ### 2. Smoothing with tau = 3 ###
    counts = moving_average(counts, gap)
    bins = moving_average(bins, gap)

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
    flag_min, goodness_min, popt_min = perform_gaussian_fit(min_id0,
        min_id1, max_ind, bins, counts, flat_m.size, gap, 'Min')

    ### 6. Find the inrterval of half height ###
    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val/2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val/2:
        half_id1 += 1

    ### 7. Try the fit between the minima and check its goodness ###
    flag_half, goodness_half, popt_half = perform_gaussian_fit(half_id0,
        half_id1, max_ind, bins, counts, flat_m.size, gap, 'Half')

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

    state = StateUni(popt[0], popt[1], popt[2])
    state.build_boundaries(NUMBER_OF_SIGMAS)

    with open(OUTPUT_FILE, 'a', encoding="utf-8") as file:
        print('\n', file=file)
        print(f'\tmu = {state.mean:.4f}, sigma = {state.sigma:.4f}, area = {state.area:.4f}')
        print(f'\tmu = {state.mean:.4f}, sigma = {state.sigma:.4f}, area = {state.area:.4f}',
            file=file)
        print('\tFit goodness = ' + str(goodness), file=file)

    ### Plot the distribution and the fitted gaussians
    y_spread = np.max(m_clean) - np.min(m_clean)
    y_lim = [np.min(m_clean) - 0.025*y_spread, np.max(m_clean) + 0.025*y_spread]
    fig, ax = plt.subplots()
    plot_histo(ax, counts, bins)
    ax.set_xlim(y_lim)
    tmp_popt = [state.mean, state.sigma, state.area/flat_m.size]
    ax.plot(np.linspace(bins[0], bins[-1], 1000),
        gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))

    fig.savefig(filename + '.png', dpi=600)
    plt.close(fig)

    return state

def find_stable_trj(
        m_clean: np.ndarray, tau_window: int, state: StateUni,
        all_the_labels: np.ndarray, offset: int
    ):
    """
    Identifies stable windows in a trajectory based on criteria.

    Args:
    - m_clean (np.ndarray): Input trajectory data.
    - tau_window (int): Size of the window for analysis.
    - state (StateUni): Object containing stable state parameters.
    - all_the_labels (np.ndarray): Labels indicating window classifications.
    - offset (int): Offset value for classifying stable windows.

    Returns:
    - m2_array (np.ndarray): Array of non-stable windows.
    - fw (float): Fraction of windows classified as stable.
    - one_last_state (bool): Indicates if there's one last state remaining.

    Notes:
    - Computes stable windows using criteria based on given state thresholds.
    - Updates the window labels to indicate stable windows with an offset.
    - Calculates the fraction of stable windows found and prints the value.
    - Returns the array of non-stable windows and related information.
    """
    print('* Finding stable windows...')

    # Calculate the number of windows in the trajectory
    number_of_windows = all_the_labels.shape[1]

    mask_unclassified = all_the_labels < 0.5
    m_reshaped = m_clean[:, :number_of_windows*tau_window].reshape(m_clean.shape[0],
        number_of_windows, tau_window)
    mask_inf = np.min(m_reshaped, axis=2) >= state.th_inf[0]
    mask_sup = np.max(m_reshaped, axis=2) <= state.th_sup[0]
    mask = mask_unclassified & mask_inf & mask_sup

    all_the_labels[mask] = offset + 1
    counter = np.sum(mask)

    # Initialize an empty list to store non-stable windows
    remaning_data = []
    mask_remaining = mask_unclassified & ~mask
    for i, window in np.argwhere(mask_remaining):
        r_w = m_clean[i, window*tau_window:(window + 1)*tau_window]
        remaning_data.append(r_w)

    # Calculate the fraction of stable windows found
    window_fraction = counter/(all_the_labels.size)

    # Print the fraction of stable windows
    with open(OUTPUT_FILE, 'a', encoding="utf-8") as file:
        print(f'\tFraction of windows in state {offset + 1} = {window_fraction:.3}')
        print(f'\tFraction of windows in state {offset + 1} = {window_fraction:.3}', file=file)

    # Convert the list of non-stable windows to a NumPy array
    m2_array = np.array(remaning_data)
    one_last_state = True
    if len(m2_array) == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows,
    # and the updated list_of_states
    return m2_array, window_fraction, one_last_state

def iterative_search(data: UniData, par: Parameters, name: str):
    """
    Performs an iterative search for stable states in a trajectory.

    Args:
    - data (UniData): Input trajectory data.
    - par (Parameters): Object containing parameters for the search.
    - name (str): Name for identifying output figures.

    Returns:
    - atl (np.ndarray): Updated labels for each window.
    - lis (list): List of identified states.
    - one_last_state (bool): Indicates if there's one last state remaining.

    Notes:
    - Divides the trajectory into windows and iteratively identifies stable states.
    - Uses Gaussian fitting and stability criteria to determine stable windows.
    - Updates labels for each window based on identified stable states.
    - Returns the updated labels, list of identified states, and a flag for one last state.
    """

    # Initialize an array to store labels for each window.
    num_windows = int(data.num_of_steps / par.tau_w)
    tmp_labels = np.zeros((data.num_of_particles, num_windows)).astype(int)

    states_list = []
    m_copy = data.matrix
    iteration_id = 1
    states_counter = 0
    one_last_state = False
    while True:
        ### Locate and fit maximum in the signal distribution
        state = gauss_fit_max(m_copy, par, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
        if state is None:
            print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')
            break

        ### Find the windows in which the trajectories are stable in the maximum
        m_next, counter, one_last_state = find_stable_trj(data.matrix, par.tau_w, state,
            tmp_labels, states_counter)
        state.perc = counter

        states_list.append(state)
        states_counter += 1
        iteration_id += 1
        ### Exit the loop if no new stable windows are found
        if counter <= 0.0:
            print('Iterations interrupted because no data point has been assigned to last state. ')
            break
        m_copy = m_next

    atl, lis = relabel_states(tmp_labels, states_list)
    return atl, lis, one_last_state

def plot_cumulative_figure(m_clean: np.ndarray, par: Parameters,
    list_of_states: list[StateUni], filename: str):
    """
    Generates a cumulative figure with signal trajectories and state Gaussian distributions.

    Args:
    - m_clean (np.ndarray): Input trajectory data.
    - par (Parameters): Object containing parameters for plotting.
    - list_of_states (list[StateUni]): List of identified states.
    - filename (str): Name for the output figure file.

    Notes:
    - Plots signal trajectories and Gaussian distributions of identified states.
    - Visualizes state thresholds and their corresponding signal ranges.
    - Saves the figure as a PNG file in the 'output_figures' directory.
    """

    print('* Printing cumulative figure...')

    # Compute histogram of flattened m_clean
    flat_m = m_clean.flatten()
    counts, bins = np.histogram(flat_m, bins=par.bins, density=True)
    counts *= flat_m.size

    # Create a 1x2 subplots with shared y-axis
    fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]},
        figsize=(9, 4.8))

    # Plot the histogram on the right subplot (ax[1])
    ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

    # Create a color palette for plotting states
    palette = []
    n_states = len(list_of_states)
    cmap = plt.get_cmap('viridis', n_states + 1)
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Define time and y-axis limits for the left subplot (ax[0])
    y_spread = np.max(m_clean) - np.min(m_clean)
    y_lim = [np.min(m_clean) - 0.025*y_spread, np.max(m_clean) + 0.025*y_spread]
    time = par.print_time(m_clean.shape[1])

    # Plot the individual trajectories on the left subplot (ax[0])
    step = 10 if m_clean.size > 1000000 else 1
    for mol in m_clean[::step]:
        ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

    # Plot the Gaussian distributions of states on the right subplot (ax[1])
    for state_id, state in enumerate(list_of_states):
        popt = [state.mean, state.sigma, state.area]
        ax[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *popt),
            np.linspace(bins[0], bins[-1], 1000), color=palette[state_id])

    # Plot the horizontal lines and shaded regions to mark states' thresholds
    style_color_map = {
        0: ('--', 'xkcd:black'),
        1: ('--', 'xkcd:blue'),
        2: ('--', 'xkcd:red'),
    }

    time2 = np.linspace(time[0] - 0.05*(time[-1] - time[0]),
        time[-1] + 0.05*(time[-1] - time[0]), 100)
    for state_id, state in enumerate(list_of_states):
        linestyle, color = style_color_map.get(state.th_inf[1], ('-', 'xkcd:black'))
        ax[1].hlines(state.th_inf[0], xmin=0.0, xmax=np.amax(counts),
            linestyle=linestyle, color=color)
        ax[0].fill_between(time2, state.th_inf[0], state.th_sup[0],
            color=palette[state_id], alpha=0.25)
    ax[1].hlines(list_of_states[-1].th_sup[0], xmin=0.0, xmax=np.amax(counts),
        linestyle=linestyle, color='black')

    # Set plot titles and axis labels
    ax[0].set_ylabel('Signal')
    ax[0].set_xlabel(r'Time $t$ ' + par.t_units)
    ax[0].set_xlim([time2[0], time2[-1]])
    ax[0].set_ylim(y_lim)
    ax[1].set_xticklabels([])

    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def plot_one_trajectory(m_clean: np.ndarray, par: Parameters,
    all_the_labels: np.ndarray, filename: str):
    """
    Plots a single trajectory of an example particle with labeled data points.

    Args:
    - m (np.ndarray): Input trajectory data.
    - par (Parameters): Object containing parameters for plotting.
    - all_the_labels (np.ndarray): Labels indicating data points' classifications.
    - filename (str): Name for the output figure file.

    Notes:
    - Plots a single trajectory with labeled data points based on classifications.
    - Uses a colormap to differentiate and visualize different data point labels.
    - Saves the figure as a PNG file in the 'output_figures' directory.
    """

    example_id = par.example_id
    # Get the signal of the example particle
    signal = m_clean[example_id][:all_the_labels.shape[1]]

    # Create time values for the x-axis
    time = par.print_time(all_the_labels.shape[1])

    # Create a figure and axes for the plot
    fig, ax = plt.subplots()

    # Create a colormap to map colors to the labels of the example particle
    unique_labels = np.unique(all_the_labels)
    # If there are no assigned window, we still need the "0" state
    # for consistency:
    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)

    cmap = plt.get_cmap('viridis',
        np.max(unique_labels) - np.min(unique_labels) + 1)
    color = all_the_labels[example_id]
    ax.plot(time, signal, c='black', lw=0.1)

    # Plot the signal as a line and scatter plot with colors based on the labels
    ax.scatter(time, signal, c=color, cmap=cmap,
        vmin=np.min(unique_labels), vmax=np.max(unique_labels), s=1.0)

    # Add title and labels to the axes
    fig.suptitle('Example particle: ID = ' + str(example_id))
    ax.set_xlabel('Time ' + par.t_units)
    ax.set_ylabel('Normalized signal')

    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def timeseries_analysis(original_data: UniData, original_par: Parameters,
    tau_w: int, t_smooth: int):
    """
    Performs an analysis pipeline on time series data.

    Args:
    - m_raw (np.ndarray): Raw input time series data.
    - par (Parameters): Object containing parameters for analysis.
    - tau_w (int): the time window for the analysis
    - t_smooth (int): the width of the moving average for the analysis

    Returns:
    - num_states (int): Number of identified states.
    - fraction_0 (float): Fraction of unclassified data points.

    Notes:
    - Prepares the data, performs an iterative search for states, and sets final states.
    - Analyzes the time series data based on specified parameters in the 'par' object.
    - Handles memory cleanup after processing to prevent accumulation.
    - Returns the number of identified states and the fraction of unclassified data points.
    """

    print('* New analysis: ', tau_w, t_smooth)
    name = str(t_smooth) + '_' + str(tau_w) + '_'

    par = original_par.create_copy()
    par.tau_w = tau_w
    par.t_smooth = t_smooth
    data = original_data.create_copy()

    data = preparing_the_data(data, par)
    plot_input_data(data, par, name + 'Fig0')

    tmp_labels, list_of_states, one_last_state = iterative_search(data, par, name)

    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        # We need to free the memory otherwise it accumulates
        del data
        del tmp_labels
        return 1, 1.0

    list_of_states, data.labels = set_final_states(list_of_states, tmp_labels, data.range)

    # We need to free the memory otherwise it accumulates
    del data
    del tmp_labels

    fraction_0 = 1 - np.sum([ state.perc for state in list_of_states ])
    if one_last_state:
        print('Number of states identified:', len(list_of_states) + 1,
            '[' + str(fraction_0) + ']\n')
        return len(list_of_states) + 1, fraction_0

    print('Number of states identified:', len(list_of_states), '[' + str(fraction_0) + ']\n')
    return len(list_of_states), fraction_0

def full_output_analysis(data: UniData, par: Parameters):
    """
    Conducts a comprehensive analysis pipeline on a dataset,
    generating multiple figures and outputs.

    Args:
    - data (UniData): Raw input data.
    - par (Parameters): Object containing parameters for analysis.

    Notes:
    - Prepares the data, conducts iterative search for states, and sets final states.
    - Computes cluster mean sequences, assigns single frames, and generates various plots.
    - Prints molecular labels and colored trajectories based on analysis results.
    """

    tau_w = par.tau_w
    data = preparing_the_data(data, par)
    plot_input_data(data, par, 'Fig0')

    tmp_labels, list_of_states, _ = iterative_search(data, par, '')
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        return
    list_of_states, data.labels = set_final_states(list_of_states, tmp_labels, data.range)

    data.plot_medoids('Fig4')
    plot_state_populations(data.labels, par, 'Fig5')
    # sankey(data.labels, [0, 10, 20, 30, 40], par, 'Fig6')
    # sankey(data.labels, [1, 53, 193], par, 'Fig6')

    all_the_labels = assign_single_frames(data.labels, tau_w)

    plot_cumulative_figure(data.matrix, par, list_of_states, 'Fig2')
    plot_one_trajectory(data.matrix, par, all_the_labels, 'Fig3')

    if os.path.exists('trajectory.xyz'):
        print_colored_trj_from_xyz('trajectory.xyz', all_the_labels, par)
    else:
        print_mol_labels_fbf_xyz(all_the_labels)

    return all_the_labels

def time_resolution_analysis(data: UniData, par: Parameters, perform_anew: bool):
    """
    Performs Temporal Resolution Analysis (TRA) to explore parameter space and analyze the dataset.

    Args:
    - data (UniData): Raw input data.
    - par (Parameters): Object containing parameters for analysis.
    - perform_anew (bool): Flag to indicate whether to perform analysis anew
        or load previous results.

    Notes:
    - Conducts TRA for different combinations of parameters.
    - Analyzes the dataset with varying 'tau_window' and 't_smooth'.
    - Saves results to text files and plots t.r.a. figures based on analysis outcomes.
    """
    tau_window_list, t_smooth_list = param_grid(par, data.num_of_steps)

    if perform_anew:
        ### If the analysis hat to be performed anew ###
        number_of_states = []
        fraction_0 = []
        for tau_w in tau_window_list:
            tmp = [tau_w]
            tmp1 = [tau_w]
            for t_s in t_smooth_list:
                n_s, f_0 = timeseries_analysis(data, par, tau_w, t_s)
                tmp.append(n_s)
                tmp1.append(f_0)
            number_of_states.append(tmp)
            fraction_0.append(tmp1)
        number_of_states_arr = np.array(number_of_states)
        fraction_0_arr = np.array(fraction_0)

        np.savetxt('number_of_states.txt', number_of_states, fmt='%i',
            delimiter='\t', header='tau_window\t number_of_states for different t_smooth')
        np.savetxt('fraction_0.txt', fraction_0, delimiter=' ',
            header='tau_window\t fraction in ENV0 for different t_smooth')
    else:
        ### Otherwise, just do this ###
        number_of_states_arr = np.loadtxt('number_of_states.txt')
        fraction_0_arr = np.loadtxt('fraction_0.txt')

    plot_tra_figure(number_of_states_arr, fraction_0_arr, par)

def main():
    """
    all_the_input_stuff() reads the data and the parameters
    time_resolution_analysis() explore the parameter (tau_window, t_smooth) space.
        Use 'False' to skip it.
    full_output_analysis() performs a detailed analysis with the chosen parameters.
    """
    data, par = all_the_input_stuff()
    time_resolution_analysis(data, par, True)
    all_the_labels = full_output_analysis(data, par)
    return all_the_labels

if __name__ == "__main__":
    main()
