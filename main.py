from functions import *

OUTPUT_FILE = 'states_output.txt'
SHOW_PLOT = False

def all_the_input_stuff():
    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters('input_parameters.txt')

    # Read raw data from the specified directory/files.
    if type(data_directory) == str:
        m_raw = read_data(data_directory)
    else:
        print('\tERROR: data_directory.txt is missing or wrongly formatted. ')

    # Remove initial frames based on 'tau_delay'.
    m_raw = m_raw[:, par.t_delay:]

    ### Create files for output
    with open(OUTPUT_FILE, 'w') as f:
        print('#', file=f)
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
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    return m_raw, par

def preparing_the_data(m_raw: np.ndarray, par: Parameters):
    tau_window, t_smooth, t_conv, t_units = par.tau_w, par.t_smooth, par.t_conv, par.t_units

    # Apply filtering on the data
    m = moving_average(m_raw, t_smooth)
    # m = np.array([ butter_lowpass_filter(x, 1/double(t_smooth), 1, 2) for x in m_raw ])

    sig_max = np.max(m)
    sig_min = np.min(m)
    ###################################################################
    ### Normalize the data to the range [0, 1]. Usually not needed. ###
    # m = (m - sig_min)/(sig_max - sig_min)
    # sig_max = np.max(m)
    # sig_min = np.min(m)
    ###################################################################

    # Get the number of particles and total frames in the trajectory.
    tot_N = m.shape[0]
    tot_T = m.shape[1]

    # Calculate the number of windows for the analysis.
    num_windows = int(tot_T / tau_window)

    # Print informative messages about trajectory details.
    print('\tTrajectory has ' + str(tot_N) + ' particles. ')
    print('\tTrajectory of length ' + str(tot_T) + ' frames (' + str(tot_T*t_conv), t_units + ')')
    print('\tUsing ' + str(num_windows) + ' windows of length ' + str(tau_window) +
        ' frames (' + str(tau_window*t_conv), t_units + ')')

    return m, [sig_min, sig_max]

def plot_input_data(m: np.ndarray, par: Parameters, filename: str):
    # Extract relevant parameters from par
    tau_window, tau_delay, t_conv, t_units = par.tau_w, par.t_delay, par.t_conv, par.t_units

    # Flatten the m matrix and compute histogram counts and bins
    flat_m = m.flatten()
    bins = par.bins
    counts, bins = np.histogram(flat_m, bins=bins, density=True)
    counts *= flat_m.size

    # Create a plot with two subplots (side-by-side)
    fig, ax = plt.subplots(1, 2, sharey=True,
        gridspec_kw={'width_ratios': [3, 1]},figsize=(9, 4.8))

    # Plot histogram in the second subplot (right side)
    ax[1].stairs(counts, bins, fill=True, orientation='horizontal')

    # Compute the time array for the x-axis of the first subplot (left side)
    time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + m.shape[1],
        m.shape[1])*t_conv

    # Plot the individual trajectories in the first subplot (left side)
    step = 10 if m.size > 1000000 else 1
    for idx, mol in enumerate(m[::step]):
        ax[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5, rasterized=True)

    # Set labels and titles for the plots
    ax[0].set_ylabel('Signal')
    ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
    ax[1].set_xticklabels([])

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def gauss_fit_max(m: np.ndarray, par: Parameters, filename: str):
    print('* Gaussian fit...')
    flat_m = m.flatten()

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
    flag_min = 1
    goodness_min = 5
    selected_bins = bins[min_id0:min_id1]
    selected_counts = counts[min_id0:min_id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[min_id0] - bins[min_id1])/6
    a0 = counts[max_ind]*np.sqrt(np.pi)*sigma0
    try:
        popt_min, pcov = scipy.optimize.curve_fit(Gaussian, selected_bins, selected_counts,
            p0=[mu0, sigma0, a0])
        if popt_min[1] < 0:
            popt_min[1] = -popt_min[1]
            popt_min[2] = -popt_min[2]
        gauss_max = popt_min[2]*np.sqrt(np.pi)*popt_min[1]
        if gauss_max < a0/2:
            goodness_min -= 1
        popt_min[2] *= flat_m.size
        if popt_min[0] < selected_bins[0] or popt_min[0] > selected_bins[-1]:
            goodness_min -= 1
        if popt_min[1] > selected_bins[-1] - selected_bins[0]:
            goodness_min -= 1
        perr = np.sqrt(np.diag(pcov))
        for j in range(len(perr)):
            if perr[j]/popt_min[j] > 0.5:
                goodness_min -= 1
        if min_id1 - min_id0 <= gap:
            goodness_min -= 1
    except RuntimeError:
        print('\tMin fit: Runtime error. ')
        flag_min = 0
    except TypeError:
        print('\tMin fit: TypeError.')
        flag_min = 0
    except ValueError:
        print('\tMin fit: ValueError.')
        flag_min = 0

    ### 6. Find the inrterval of half height ###
    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val/2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val/2:
        half_id1 += 1

    ### 7. Try the fit between the minima and check its goodness ###
    flag_half = 1
    goodness_half = 5
    selected_bins = bins[half_id0:half_id1]
    selected_counts = counts[half_id0:half_id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[half_id0] - bins[half_id1])/6
    a0 = counts[max_ind]*np.sqrt(np.pi)*sigma0
    try:
        popt_half, pcov = scipy.optimize.curve_fit(Gaussian, selected_bins, selected_counts,
            p0=[mu0, sigma0, a0])
        if popt_half[1] < 0:
            popt_half[1] = -popt_half[1]
            popt_half[2] = -popt_half[2]
        gauss_max = popt_half[2]*np.sqrt(np.pi)*popt_half[1]
        if gauss_max < a0/2:
            goodness_half -= 1
        popt_half[2] *= flat_m.size
        if popt_half[0] < selected_bins[0] or popt_half[0] > selected_bins[-1]:
            goodness_half -= 1
        if popt_half[1] > selected_bins[-1] - selected_bins[0]:
            goodness_half -= 1
        perr = np.sqrt(np.diag(pcov))
        for j in range(len(perr)):
            if perr[j]/popt_half[j] > 0.5:
                goodness_half -= 1
        if min_id1 - min_id0 < gap:
            goodness_half -= 1
    except RuntimeError:
        print('\tHalf fit: Runtime error. ')
        flag_half = 0
    except TypeError:
        print('\tHalf fit: TypeError.')
        flag_half = 0
    except ValueError:
        print('\tHalf fit: ValueError.')
        flag_half = 0

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

    state = State(popt[0], popt[1], popt[2])

    with open(OUTPUT_FILE, 'a') as f:
        print('\n', file=f)
        print(f'\tmu = {state.mu:.4f}, sigma = {state.sigma:.4f}, area = {state.area:.4f}')
        print(f'\tmu = {state.mu:.4f}, sigma = {state.sigma:.4f}, area = {state.area:.4f}', file=f)
        print('\tFit goodness = ' + str(goodness), file=f)

    ### Plot the distribution and the fitted Gaussians
    y_lim = [np.min(m) - 0.025*(np.max(m) - np.min(m)), np.max(m) + 0.025*(np.max(m) - np.min(m))]
    fig, ax = plt.subplots()
    plot_histo(ax, counts, bins)
    ax.set_xlim(y_lim)
    tmp_popt = [state.mu, state.sigma, state.area/flat_m.size]
    ax.plot(np.linspace(bins[0], bins[-1], 1000),
        Gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))

    if SHOW_PLOT:
        plt.show()
    fig.savefig(filename + '.png', dpi=600)
    plt.close(fig)

    return state

def find_stable_trj(m: np.ndarray, tau_window: int, state: State, all_the_labels: np.ndarray, offset: int):
    print('* Finding stable windows...')

    # Calculate the number of windows in the trajectory
    number_of_windows = all_the_labels.shape[1]

    mask_unclassified = all_the_labels < 0.5
    m_reshaped = m[:, :number_of_windows*tau_window].reshape(m.shape[0],
        number_of_windows, tau_window)
    mask_inf = np.min(m_reshaped, axis=2) >= state.th_inf[0]
    mask_sup = np.max(m_reshaped, axis=2) <= state.th_sup[0]
    mask = mask_unclassified & mask_inf & mask_sup

    all_the_labels[mask] = offset + 1
    counter = np.sum(mask)

    # Initialize an empty list to store non-stable windows
    m2 = []
    mask_remaining = mask_unclassified & ~mask
    for i, w in np.argwhere(mask_remaining):
        r_w = m[i, w*tau_window:(w + 1)*tau_window]
        m2.append(r_w)

    # Calculate the fraction of stable windows found
    fw = counter/(all_the_labels.size)

    # Print the fraction of stable windows
    with open(OUTPUT_FILE, 'a') as f:
        print(f'\tFraction of windows in state {offset + 1} = {fw:.3}')
        print(f'\tFraction of windows in state {offset + 1} = {fw:.3}', file=f)

    # Convert the list of non-stable windows to a NumPy array
    m2_array = np.array(m2)
    one_last_state = True
    if len(m2_array) == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows,
    # and the updated list_of_states
    return m2_array, fw, one_last_state

def iterative_search(m: np.ndarray, par: Parameters, name: str):
    tau_w, bins = par.tau_w, par.bins

    # Initialize an array to store labels for each window.
    num_windows = int(m.shape[1] / tau_w)
    all_the_labels = np.zeros((m.shape[0], num_windows))

    states_list = []
    m1 = m
    iteration_id = 1
    states_counter = 0
    one_last_state = False
    while True:
        ### Locate and fit maximum in the signal distribution
        state = gauss_fit_max(m1, par, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
        if state == None:
            print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')
            break

        ### Find the windows in which the trajectories are stable in the maximum
        m2, c, one_last_state = find_stable_trj(m, tau_w, state, all_the_labels, states_counter)
        state.perc = c

        states_list.append(state)
        states_counter += 1
        iteration_id += 1
        ### Exit the loop if no new stable windows are found
        if c <= 0.0:
            print('Iterations interrupted because no data point has been assigned to last state. ')
            break
        else:
            m1 = m2

    atl, lis = relabel_states(all_the_labels, states_list)
    return atl, lis, one_last_state

def plot_cumulative_figure(m: np.ndarray, par: Parameters, list_of_states: list[State], filename: str):
    print('* Printing cumulative figure...')
    tau_window, tau_delay, t_conv, t_units, bins = par.tau_w, par.t_delay, par.t_conv, par.t_units, par.bins
    n_states = len(list_of_states)

    # Compute histogram of flattened m
    flat_m = m.flatten()
    counts, bins = np.histogram(flat_m, bins=bins, density=True)
    counts *= flat_m.size

    # Create a 1x2 subplots with shared y-axis
    fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

    # Plot the histogram on the right subplot (ax[1])
    ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

    # Create a color palette for plotting states
    palette = []
    cmap = plt.get_cmap('viridis', n_states + 1)
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Define time and y-axis limits for the left subplot (ax[0])
    y_lim = [np.min(m) - 0.025*(np.max(m) - np.min(m)), np.max(m) + 0.025*(np.max(m) - np.min(m))]
    time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + m.shape[1], m.shape[1])*t_conv

    # Plot the individual trajectories on the left subplot (ax[0])
    step = 10 if m.size > 1000000 else 1
    for idx, mol in enumerate(m[::step]):
        ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

    # Plot the Gaussian distributions of states on the right subplot (ax[1])
    for s in range(n_states):
        popt = [list_of_states[s].mu, list_of_states[s].sigma, list_of_states[s].area]
        ax[1].plot(Gaussian(np.linspace(bins[0], bins[-1], 1000), *popt), np.linspace(bins[0], bins[-1], 1000), color=palette[s])

    # Plot the horizontal lines and shaded regions to mark states' thresholds
    style_color_map = {
        0: ('--', 'xkcd:black'),
        1: ('--', 'xkcd:blue'),
        2: ('--', 'xkcd:red'),
    }

    time2 = np.linspace(time[0] - 0.05*(time[-1] - time[0]), time[-1] + 0.05*(time[-1] - time[0]), 100)
    for n, state in enumerate(list_of_states):
        linestyle, color = style_color_map.get(state.th_inf[1], ('-', 'xkcd:black'))
        ax[1].hlines(state.th_inf[0], xmin=0.0, xmax=np.amax(counts), linestyle=linestyle, color=color)
        ax[0].fill_between(time2, state.th_inf[0], state.th_sup[0], color=palette[n], alpha=0.25)
    ax[1].hlines(list_of_states[-1].th_sup[0], xmin=0.0, xmax=np.amax(counts), linestyle=linestyle, color='black')

    # Set plot titles and axis labels
    ax[0].set_ylabel('Signal')
    ax[0].set_xlabel(r'Time $t$ ' + t_units)
    ax[0].set_xlim([time2[0], time2[-1]])
    ax[0].set_ylim(y_lim)
    ax[1].set_xticklabels([])

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def plot_one_trajectory(m: np.ndarray, par: Parameters, all_the_labels: np.ndarray, filename: str):
    tau_window, tau_delay, t_conv, t_units, example_ID = par.tau_w, par.t_delay, par.t_conv, par.t_units, par.example_ID
    # Get the signal of the example particle
    signal = m[example_ID][:all_the_labels.shape[1]]

    # Create time values for the x-axis
    times = np.arange(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + m.shape[1]) * t_conv
    times = times[:all_the_labels.shape[1]]

    # Create a figure and axes for the plot
    fig, ax = plt.subplots()

    # Create a colormap to map colors to the labels of the example particle
    cmap = plt.get_cmap('viridis', np.max(np.unique(all_the_labels)) - np.min(np.unique(all_the_labels)) + 1)
    color = all_the_labels[example_ID]
    ax.plot(times, signal, c='black', lw=0.1)

    # Plot the signal as a line and scatter plot with colors based on the labels
    ax.scatter(times, signal, c=color, cmap=cmap, vmin=np.min(np.unique(all_the_labels)), vmax=np.max(np.unique(all_the_labels)), s=1.0)

    # Add title and labels to the axes
    fig.suptitle('Example particle: ID = ' + str(example_ID))
    ax.set_xlabel('Time ' + t_units)
    ax.set_ylabel('Normalized signal')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def timeseries_analysis(m_raw: np.ndarray, par: Parameters):
    tau_w, t_smooth = par.tau_w, par.t_smooth
    name = str(t_smooth) + '_' + str(tau_w) + '_'
    m, m_range = preparing_the_data(m_raw, par)
    plot_input_data(m, par, name + 'Fig0')

    all_the_labels, list_of_states, one_last_state = iterative_search(m, par, name)

    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        # We need to free the memory otherwise it accumulates
        del m_raw
        del m
        del all_the_labels
        return 1, 1.0

    list_of_states, all_the_labels = set_final_states(list_of_states, all_the_labels, m_range)

    # We need to free the memory otherwise it accumulates
    del m_raw
    del m
    del all_the_labels

    fraction_0 = 1 - np.sum([ state.perc for state in list_of_states ])
    if one_last_state:
        return len(list_of_states) + 1, fraction_0
    else:
        return len(list_of_states), fraction_0

def compute_cluster_mean_seq(m: np.ndarray, all_the_labels: np.ndarray, tau_window: int):
    # Initialize lists to store cluster means and standard deviations
    center_list = []
    std_list = []

    # Loop through unique labels (clusters)
    for label in np.unique(all_the_labels):
        tmp = []
        # Iterate through molecules and their labels
        for i, mol in enumerate(all_the_labels):
            for w, l in enumerate(mol):
                 # Define time interval
                t0 = w*tau_window
                t1 = (w + 1)*tau_window
                # If the label matches the current cluster, append the corresponding data to tmp
                if l == label:
                    tmp.append(m[i][t0:t1])

        # Calculate mean and standard deviation for the current cluster
        center_list.append(np.mean(tmp, axis=0))
        std_list.append(np.std(tmp, axis=0))

    # Create a color palette
    palette = []
    cmap = plt.get_cmap('viridis', np.unique(all_the_labels).size)
    palette.append(rgb2hex(cmap(0)))
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Plot
    fig, ax = plt.subplots()
    x = range(tau_window)
    for l, center in enumerate(center_list):
        err_inf = center - std_list[l]
        err_sup = center + std_list[l]
        ax.fill_between(x, err_inf, err_sup, alpha=0.25, color=palette[l])
        ax.plot(x, center, label='ENV'+str(l), marker='o', c=palette[l])
    fig.suptitle('Average time sequence inside each environments')
    ax.set_xlabel(r'Time $t$ [frames]')
    ax.set_ylabel(r'Signal')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/Fig4.png', dpi=600)

def full_output_analysis(m_raw: np.ndarray, par: Parameters):
    tau_w = par.tau_w
    m, m_range = preparing_the_data(m_raw, par)
    plot_input_data(m, par, 'Fig0')

    all_the_labels, list_of_states, one_last_state = iterative_search(m, par, '')
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        return
    list_of_states, all_the_labels = set_final_states(list_of_states, all_the_labels, m_range)

    compute_cluster_mean_seq(m, all_the_labels, tau_w)

    all_the_labels = assign_single_frames(all_the_labels, tau_w)

    plot_cumulative_figure(m, par, list_of_states, 'Fig2')
    plot_one_trajectory(m, par, all_the_labels, 'Fig3')
    # sankey(all_the_labels, [0, 100, 200, 300], par, 'Fig5', SHOW_PLOT)
    plot_state_populations(all_the_labels, par, 'Fig5', SHOW_PLOT)

    print_mol_labels_fbf_xyz(all_the_labels)
    print_colored_trj_from_xyz('trajectory.xyz', all_the_labels, par)

def TRA_analysis(m_raw: np.ndarray, par: Parameters, perform_anew: bool):
    tau_window_list, t_smooth_list = param_grid(par, m_raw.shape[1])

    if perform_anew:
        ### If the analysis hat to be performed anew ###
        number_of_states = []
        fraction_0 = []
        for tau_w in tau_window_list:
            tmp = [tau_w]
            tmp1 = [tau_w]
            for t_s in t_smooth_list:
                print('\n* New analysis: ', tau_w, t_s)
                tmp_par = copy.deepcopy(par)
                tmp_par.tau_w = tau_w
                tmp_par.t_smooth = t_s
                n_s, f0 = timeseries_analysis(m_raw, tmp_par)
                tmp.append(n_s)
                tmp1.append(f0)
            number_of_states.append(tmp)
            fraction_0.append(tmp1)
        number_of_states_arr = np.array(number_of_states)
        fraction_0_arr = np.array(fraction_0)
        header = 'tau_window t_s = 1 t_s = 2 t_s = 3 t_s = 4 t_s = 5'
        np.savetxt('number_of_states.txt', number_of_states, fmt='%i', delimiter='\t', header=header)
        np.savetxt('fraction_0.txt', fraction_0, delimiter=' ', header=header)
    else:
        ### Otherwise, just do this ###
        number_of_states_arr = np.loadtxt('number_of_states.txt')
        fraction_0_arr = np.loadtxt('fraction_0.txt')

    plot_TRA_figure(number_of_states_arr, fraction_0_arr, par, SHOW_PLOT)

def main():
    m_raw, par = all_the_input_stuff()
    TRA_analysis(m_raw, par, False)
    full_output_analysis(m_raw, par)

if __name__ == "__main__":
    main()
