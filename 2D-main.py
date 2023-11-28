from functions import *

OUTPUT_FILE = 'states_output.txt'
SHOW_PLOT = False

def all_the_input_stuff():
    # Read input parameters from files.
    data_directory = read_input_data()
    par = Parameters('input_parameters.txt')

    tmp_m_raw = []
    for d in range(len(data_directory)):
        # Read raw data from the specified directory/files.
        m_raw = read_data(data_directory[d])

        # Remove initial frames based on 'tau_delay'.
        m_raw = m_raw[:, par.t_delay:]

        tmp_m_raw.append(m_raw)

    for d in range(len(tmp_m_raw) - 1):
        if tmp_m_raw[d].shape != tmp_m_raw[d + 1].shape :
            print('ERROR: The signals do not correspond. Abort.')
            return

    ### Create files for output
    with open(OUTPUT_FILE, 'w') as f:
        f.write('#')
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

    # Return required data for further analysis.
    return tmp_m_raw, par

def preparing_the_data(tmp_m_raw: np.ndarray, par: Parameters):
    tau_window, t_smooth, t_conv, t_units = par.tau_w, par.t_smooth, par.t_conv, par.t_units

    m = []
    for d, m_raw in enumerate(tmp_m_raw):
        # Apply filtering on the data
        tmp_m = moving_average(m_raw, t_smooth)

        # Normalize the data to the range [0, 1].
        sig_max = np.max(tmp_m)
        sig_min = np.min(tmp_m)
        # m = (m - sig_min)/(sig_max - sig_min)

        m.append(tmp_m)

    m_arr = np.array(m)
    m_limits = [ [np.min(x), np.max(x) ] for x in m_arr ]
    m_arr = np.transpose(m_arr, axes=(1, 2, 0))

    total_particles = m_arr.shape[0]
    total_time = m_arr.shape[1]
    # Calculate the number of windows for the analysis.
    num_windows = int(total_time / tau_window)

    # Print informative messages about trajectory details.
    print('\tTrajectory has ' + str(total_particles) + ' particles. ')
    print('\tTrajectory of length ' + str(total_time) + ' frames (' + str(total_time*t_conv) + ' ' + t_units + ').')
    print('\tUsing ' + str(num_windows) + ' windows of length ' + str(tau_window) + ' frames (' + str(tau_window*t_conv) + ' ' + t_units + ').')

    return m_arr, m_limits

def plot_input_data(m: np.ndarray, par: Parameters, filename: str):
    tau_window, tau_delay, t_conv, t_units, bins = par.tau_w, par.t_delay, par.t_conv, par.t_units, par.bins
    bin_selection = []
    counts_selection = []
    for d in range(m.shape[2]):
        # Flatten the m matrix and compute histogram counts and bins
        flat_m = m[:,:,d].flatten()
        counts0, bins0 = np.histogram(flat_m, bins=bins, density=True)
        counts0 *= flat_m.size
        bin_selection.append(bins0)
        counts_selection.append(counts0)

    if m.shape[2] == 2:
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
        for idx, mol in enumerate(m):
            if np.max(mol) == np.max(m):
                id_max = idx
            if np.min(mol) == np.min(m):
                id_min = idx
        step = 10 if m.size > 1000000 else 1
        for idx, mol in enumerate(m[::step]):
            ax2.plot(mol[:,0], mol[:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)
        ax2.plot(m[id_min][:,0], m[id_min][:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)
        ax2.plot(m[id_max][:,0], m[id_max][:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)

        # Set labels and titles for the plots
        ax2.set_ylabel('Signal 1')
        ax2.set_xlabel('Signal 2')

    elif m.shape[2] == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        
        # Plot the individual trajectories
        step = 1 if m.size > 1000000 else 1
        for idx, mol in enumerate(m[::step]):
            ax.plot(mol[:,0], mol[:,1], mol[:,2], color='black', marker='o', ms=0.5, lw=0.2, alpha=1.0, rasterized=True)

        # Set labels and titles for the plots
        ax.set_xlabel('Signal 1')
        ax.set_ylabel('Signal 2')
        ax.set_zlabel('Signal 3')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def gauss_fit_max(m: np.ndarray, m_limits: list[list[int]], bins: Union[int, str], filename: str):
    print('* Gaussian fit...')
    flat_m = m.reshape((m.shape[0]*m.shape[1], m.shape[2]))

    ### 1. Histogram with 'auto' binning ###
    if bins == 'auto':
        bins = max(int(np.power(m.size, 1/3)*2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)
    gap = 1
    if np.all([e.size > 40 for e in edges]):
        gap = 3

    ### 2. Smoothing with tau = 3 ###
    counts = moving_average_2D(counts, gap)

    ### 3. Find the maximum ###
    def find_max_index(data):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    max_val = counts.max()
    max_ind = find_max_index(counts)

    ### 4. Find the minima surrounding it ###
    def find_minima_around_max(data, max_ind, gap):
        minima = []

        for dim in range(data.ndim):
            min_id0 = max(max_ind[dim] - gap, 0)
            min_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

            tmp_max1 = copy.deepcopy(max_ind)
            tmp_max2 = copy.deepcopy(max_ind)

            tmp_max1[dim] = min_id0
            tmp_max2[dim] = min_id0 - 1
            while min_id0 > 0 and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]:
                tmp_max1[dim] -= 1
                tmp_max2[dim] -= 1
                min_id0 -= 1

            tmp_max1 = copy.deepcopy(max_ind)
            tmp_max2 = copy.deepcopy(max_ind)

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
    for dim in range(m.shape[2]):
        try:
            flag_min, goodness, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap, m_limits)
            popt[2] *= flat_m.T[0].size
            popt_min.extend(popt)
            goodness_min += goodness
        except:
            popt_min = []
            flag_min = False
            goodness_min -= 5

    ### 6. Find the interval of half height ###
    def find_half_height_around_max(data, max_ind, gap):
        max_val = data.max()
        minima = []

        for dim in range(data.ndim):
            half_id0 = max(max_ind[dim] - gap, 0)
            half_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

            tmp_max = copy.deepcopy(max_ind)

            tmp_max[dim] = half_id0
            while half_id0 > 0 and data[tuple(tmp_max)] > max_val/2:
                tmp_max[dim] -= 1
                half_id0 -= 1

            tmp_max = copy.deepcopy(max_ind)

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
    for dim in range(m.shape[2]):
        try:
            flag_half, goodness, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap, m_limits)
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

    if len(popt) != m.shape[2]*3:
        print('\tWARNING: this fit is not converging.')
        return None     

    ### Find the tresholds for state identification
    mu, sigma, area = [], [], []
    for dim in range(m.shape[2]):
        mu.append(popt[3*dim])
        sigma.append(popt[3*dim + 1])
        area.append(popt[3*dim + 2])
    state = StateMulti(np.array(mu), np.array(sigma), np.array(area))

    ### Plot the distribution and the fitted Gaussians
    if m.shape[2] == 2:
        with open(OUTPUT_FILE, 'a') as f:
            print('\n', file=f)
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}')
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}', file=f)
            print('\tFit goodness = ' + str(goodness), file=f)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = NonUniformImage(ax, interpolation='nearest')
        xcenters = (edges[0][:-1] + edges[0][1:]) / 2
        ycenters = (edges[1][:-1] + edges[1][1:]) / 2
        im.set_data(xcenters, ycenters, counts.T)
        ax.add_image(im)
        ax.scatter(mu[0], mu[1], s=8.0, c='red')
        circle1 = Ellipse(tuple(mu), sigma[0], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple(mu), state.axis[0], state.axis[1], color='r', fill=False)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.set_xlim(m_limits[0][0], m_limits[0][1])
        ax.set_ylim(m_limits[1][0], m_limits[1][1])
    elif m.shape[2] == 3:
        with open(OUTPUT_FILE, 'a') as f:
            print('\n', file=f)
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}')
            print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}', file=f)
            print('\tFit goodness = ' + str(goodness), file=f)

        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        xcenters = (edges[0][:-1] + edges[0][1:]) / 2
        ycenters = (edges[1][:-1] + edges[1][1:]) / 2
        zcenters = (edges[2][:-1] + edges[2][1:]) / 2

        im = NonUniformImage(ax[0][0], interpolation='nearest')
        im.set_data(xcenters, ycenters, np.sum(counts, axis=0))
        ax[0][0].add_image(im)
        ax[0][0].scatter(mu[0], mu[1], s=8.0, c='red')
        circle1 = Ellipse(tuple([mu[0], mu[1]]), sigma[0], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple([mu[0], mu[1]]), state.axis[0], state.axis[1], color='r', fill=False)
        ax[0][0].add_patch(circle1)
        ax[0][0].add_patch(circle2)

        im = NonUniformImage(ax[0][1], interpolation='nearest')
        im.set_data(zcenters, ycenters, np.sum(counts, axis=1))
        ax[0][1].add_image(im)
        ax[0][1].scatter(mu[2], mu[1], s=8.0, c='red')
        circle1 = Ellipse(tuple([mu[2], mu[1]]), sigma[2], sigma[1], color='r', fill=False)
        circle2 = Ellipse(tuple([mu[2], mu[1]]), state.axis[2], state.axis[1], color='r', fill=False)
        ax[0][1].add_patch(circle1)
        ax[0][1].add_patch(circle2)

        im = NonUniformImage(ax[1][0], interpolation='nearest')
        im.set_data(xcenters, zcenters, np.sum(counts, axis=2))
        ax[1][0].add_image(im)
        ax[1][0].scatter(mu[0], mu[2], s=8.0, c='red')
        circle1 = Ellipse(tuple([mu[0], mu[2]]), sigma[0], sigma[2], color='r', fill=False)
        circle2 = Ellipse(tuple([mu[0], mu[2]]), state.axis[0], state.axis[2], color='r', fill=False)
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

def find_stable_trj(m: np.ndarray, tau_window: int, state: StateMulti, all_the_labels: np.ndarray, offset: int):
    print('* Finding stable windows...')

    # Calculate the number of windows in the trajectory
    number_of_windows = int(m.shape[1] / tau_window )

    mask_unclassified = all_the_labels < 0.5
    m_reshaped = m[:, :number_of_windows*tau_window].reshape(m.shape[0], number_of_windows, tau_window, m.shape[2])
    shifted = m_reshaped - state.mean
    rescaled = shifted / state.axis
    squared_distances = np.sum(rescaled**2, axis=3)
    mask_dist = np.max(squared_distances, axis=2) <= 1.0
    mask = mask_unclassified & mask_dist

    all_the_labels[mask] = offset + 1   # Label the stable windows in the new state
    counter = np.sum(mask)              # The number of stable windows found

    # Store non-stable windows in a list, for the next iteration
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
    m2_arr = np.array(m2)
    one_last_state = True
    if len(m2_arr) == 0:
        one_last_state = False

    # Return the array of non-stable windows, the fraction of stable windows, and the updated list_of_states
    return m2_arr, fw, one_last_state

def iterative_search(m: np.ndarray, m_limits: list[list[int]], par: Parameters, name: str):
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
        state = gauss_fit_max(m1, m_limits, bins, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
        if state == None:
            print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')           
            break

        ### Find the windows in which the trajectories are stable in the maximum
        m2, c, one_last_state = find_stable_trj(m, tau_w, state, all_the_labels, states_counter)
        state.perc = c

        if c > 0.0:
            states_list.append(state)
        
        states_counter += 1
        iteration_id += 1
        ### Exit the loop if no new stable windows are found
        if c <= 0.0:
            print('Iterations interrupted because no data points have been assigned to the last state. ')
            break
        elif m2.size == 0:
            print('Iterations interrupted because all data points have been assigned to one state. ')
            break
        else:
            m1 = m2

    all_the_labels, list_of_states = relabel_states_2D(all_the_labels, states_list)
    return all_the_labels, list_of_states, one_last_state

def plot_cumulative_figure(m: np.ndarray, par: Parameters, all_the_labels: np.ndarray, list_of_states: list[StateMulti], filename: str):
    print('* Printing cumulative figure...')
    colormap = 'viridis'
    n_states = len(list_of_states) + 1
    x = plt.get_cmap(colormap, n_states)
    colors_from_cmap = x(np.arange(0, 1, 1/n_states))
    colors_from_cmap[-1] = x(1.0)
    
    fig = plt.figure(figsize=(6, 6))
    if m.shape[2] == 3:
        ax = plt.axes(projection='3d')

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m):
            if np.max(mol) == np.max(m):
                id_max = idx
            if np.min(mol) == np.min(m):
                id_min = idx

        lw = 0.05

        step = 5 if m.size > 1000000 else 1
        max_T = all_the_labels.shape[1]
        for i, mol in enumerate(m[::step]):
            ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
            c = [ int(l) for l in all_the_labels[i*step] ]
            ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, size=0.5, rasterized=True)
        
        c = [ int(l) for l in all_the_labels[id_min] ]
        ax.plot(m[id_min].T[0,:max_T], m[id_min].T[1,:max_T], m[id_min].T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m[id_min].T[0,:max_T], m[id_min].T[1,:max_T], m[id_min].T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, size=0.5, rasterized=True)
        c = [ int(l) for l in all_the_labels[id_max] ]
        ax.plot(m[id_max].T[0,:max_T], m[id_max].T[1,:max_T], m[id_max].T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m[id_max].T[0,:max_T], m[id_max].T[1,:max_T], m[id_max].T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, size=0.5, rasterized=True)

        # Plot the Gaussian distributions of states
        for s_id, S in enumerate(list_of_states):
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = S.axis[0]*np.outer(np.cos(u), np.sin(v)) + S.mean[0]
            y = S.axis[1]*np.outer(np.sin(u), np.sin(v)) + S.mean[1]
            z = S.axis[2]*np.outer(np.ones_like(u), np.cos(v)) + S.mean[2]
            ax.plot_surface(x, y, z, alpha=0.25, color=colors_from_cmap[s_id+1])

        # Set plot titles and axis labels
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
    elif m.shape[2] == 2:
        ax = plt.axes()

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m):
            if np.max(mol) == np.max(m):
                id_max = idx
            if np.min(mol) == np.min(m):
                id_min = idx

        lw = 0.05

        step = 5 if m.size > 1000000 else 1
        max_T = all_the_labels.shape[1]
        for i, mol in enumerate(m[::step]):
            ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
            c = [ int(l) for l in all_the_labels[i*step] ]
            ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

        c = [ int(l) for l in all_the_labels[id_min] ]
        ax.plot(m[id_min].T[0,:max_T], m[id_min].T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m[id_min].T[0,:max_T], m[id_min].T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
        c = [ int(l) for l in all_the_labels[id_max] ]
        ax.plot(m[id_max].T[0,:max_T], m[id_max].T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
        ax.scatter(m[id_max].T[0,:max_T], m[id_max].T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

        # Plot the Gaussian distributions of states
        for s_id, S in enumerate(list_of_states):
            ellipse = Ellipse(tuple(S.mean), S.axis[0], S.axis[1], color='black', fill=False)
            ax.add_patch(ellipse)

        # Set plot titles and axis labels
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def plot_one_trajectory(m: np.ndarray, par: Parameters, all_the_labels: np.ndarray, filename: str):
    colormap = 'viridis'
    tau_window, tau_delay, t_conv, t_units, example_id = par.tau_w, par.t_delay, par.t_conv, par.t_units, par.example_id

    # Get the signal of the example particle
    signal_x = m[example_id].T[0][:all_the_labels.shape[1]]
    signal_y = m[example_id].T[1][:all_the_labels.shape[1]]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a colormap to map colors to the labels of the example particle
    cmap = plt.get_cmap(colormap, int(np.max(np.unique(all_the_labels)) - np.min(np.unique(all_the_labels)) + 1))
    color = all_the_labels[example_id]
    ax.plot(signal_x, signal_y, c='black', lw=0.1)

    ax.scatter(signal_x, signal_y, c=color, cmap=cmap, vmin=np.min(np.unique(all_the_labels)), vmax=np.max(np.unique(all_the_labels)), s=1.0, zorder=10)

    # Set plot titles and axis labels
    fig.suptitle('Example particle: ID = ' + str(example_id))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/' + filename + '.png', dpi=600)
    plt.close(fig)

def timeseries_analysis(m_raw: np.ndarray, par: Parameters):
    tau_w, t_smooth = par.tau_w, par.t_smooth
    name = str(t_smooth) + '_' + str(tau_w) + '_'
    m, m_limits = preparing_the_data(m_raw, par)
    plot_input_data(m, par, name + 'Fig0')

    all_the_labels, list_of_states, one_last_state = iterative_search(m, m_limits, par, name)
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        # We need to free the memory otherwise it accumulates
        del m_raw
        del m
        del all_the_labels
        return 1, 1.0
    
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
    if m.shape[2] > 2:
        return

    # Initialize lists to store cluster means and standard deviations
    center_list = []

    # Loop through unique labels (clusters)
    for L in np.unique(all_the_labels):
        tmp = []
        # Iterate through molecules and their labels
        for i, mol in enumerate(all_the_labels):
            for w, l in enumerate(mol):
                # Define time interval
                t0 = w*tau_window
                t1 = (w + 1)*tau_window
                # If the label matches the current cluster, append the corresponding data to tmp
                if l == L:
                    tmp.append(m[i][t0:t1])
    
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
    for l, center in enumerate(center_list):
        x = center[:, 0]
        y = center[:, 1]
        ax.plot(x, y, label='ENV'+str(l), marker='o', c=palette[l])
    fig.suptitle('Average time sequence inside each environments')
    ax.set_xlabel(r'Signal 1')
    ax.set_ylabel(r'Signal 2')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    if SHOW_PLOT:
        plt.show()
    fig.savefig('output_figures/Fig4.png', dpi=600)

def full_output_analysis(m_raw: np.ndarray, par: Parameters):
    tau_w = par.tau_w
    m, m_limits = preparing_the_data(m_raw, par)
    plot_input_data(m, par, 'Fig0')

    all_the_labels, list_of_states, one_last_state = iterative_search(m, m_limits, par, '')
    if len(list_of_states) == 0:
        print('* No possible classification was found. ')
        return

    compute_cluster_mean_seq(m, all_the_labels, tau_w)
    all_the_labels = assign_single_frames(all_the_labels, tau_w)
    
    plot_cumulative_figure(m, par, all_the_labels, list_of_states, 'Fig2')
    plot_one_trajectory(m, par, all_the_labels, 'Fig3')
    # sankey(all_the_labels, [0, 1000, 2000, 3000], par, 'Fig5', SHOW_PLOT)
    plot_state_populations(all_the_labels, par, 'Fig5', SHOW_PLOT)

    print_mol_labels_fbf_xyz(all_the_labels)
    print_signal_with_labels(m, all_the_labels)
    print_colored_trj_from_xyz('trajectory.xyz', all_the_labels, par)

def TRA_analysis(m_raw: np.ndarray, par: Parameters, perform_anew: bool):
    tau_window_list, t_smooth_list = param_grid(par, m_raw[0].shape[1])
    
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
    """
    all_the_input_stuff() reads the data and the parameters
    time_resolution_analysis() explore the parameter (tau_window, t_smooth) space.
        Use 'False' to skip it.
    full_output_analysis() performs a detailed analysis with the chosen parameters.
    """
    m_raw, par = all_the_input_stuff()
    TRA_analysis(m_raw, par, True)
    full_output_analysis(m_raw, par)

if __name__ == "__main__":
    main()
