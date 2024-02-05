def butter_lowpass_filter(x: np.ndarray, cutoff: float, fs: float, order: int):
    nyq = 0.5
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, x)
    return y

def Savgol_filter(m: np.ndarray, window: int):
    # Step 1: Set the polynomial order for the Savitzky-Golay filter.
    poly_order = 2

    # Step 2: Apply the Savitzky-Golay filter to each row (x) in the input data matrix 'm'.
    # The result is stored in a temporary array 'tmp'.
    # 'window' is the window size for the filter.
    tmp = np.array([scipy.signal.savgol_filter(x, window, poly_order) for x in m])

    # Step 3: Since the Savitzky-Golay filter operates on a sliding window,
    # it introduces edge artifacts at the beginning and end of each row.
    # To remove these artifacts, the temporary array 'tmp' is sliced to remove the unwanted edges.
    # The amount of removal on each side is half of the 'window' value, converted to an integer.
    return tmp[:, int(window/2):-int(window/2)]


def normalize_array(x: np.ndarray):
    # Step 1: Calculate the mean value and the standard deviation of the input array 'x'.
    mean = np.mean(x)
    stddev = np.std(x)

    # Step 2: Create a temporary array 'tmp' containing the normalized version of 'x'.
    # To normalize, subtract the mean value from each element of 'x'
    # and then divide by the standard deviation.
    # This centers the data around zero (mean) and scales it based on the standard deviation.
    tmp = (x - mean) / stddev

    # Step 3: Return the normalized array 'tmp',
    # along with the calculated mean and standard deviation.
    # The returned values can be useful for further processing
    # or to revert the normalization if needed.
    return tmp, mean, stddev

def sigmoidal(x: float, a: float, b: float, alpha: float):
    return b + a/(1 + np.exp(x*alpha))

def gaussian_2d(r_points: np.ndarray, x_mean: float, y_mean: float,
    sigmax: float, sigmay: float, area: float):
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

def gaussian_full(r: np.ndarray, mx: float, my: float, sigmax: float, sigmay: float, sigmaxy: float, area: float):
    # "m" is the Gaussians' mean value (2d array)
    # "sigma" is the Gaussians' standard deviation matrix
    # "area" is the Gaussian area
    r[0] -= mx
    r[1] -= my
    arg = (r[0]/sigmax)**2 + (r[1]/sigmay)**2 + 2*r[0]*r[1]/sigmaxy**2
    norm = np.pi*sigmax*sigmay/np.sqrt(1 - (sigmax*sigmay/sigmaxy**2)**2)
    gauss = np.exp(-arg)*area/norm
    return gauss.ravel()

def fit_2D(max_ind: list[int], minima: list[int], xedges: np.ndarray, yedges: np.ndarray, counts: np.ndarray, gap: int):
    # Initialize flag and goodness variables
    flag = 1
    goodness = 11

    # Extract relevant data within the specified minima
    x_edges = xedges[minima[0]:minima[1]]
    y_edges = yedges[minima[2]:minima[3]]
    counts_selection = counts[minima[0]:minima[1],minima[2]:minima[3]]

    # Initial parameter guesses
    mux0 = xedges[max_ind[0]]
    muy0 = yedges[max_ind[1]]
    sigmax0 = (xedges[minima[1]] - xedges[minima[0]])/3
    sigmay0 = (yedges[minima[3]] - yedges[minima[2]])/3
    a0 = counts[max_ind[0]][max_ind[1]]

    # Create a meshgrid for fitting
    x, y = np.meshgrid(x_edges, y_edges)
    try:
        # Attempt to fit a 2D Gaussian using curve_fit
        popt, pcov = scipy.optimize.curve_fit(gaussian_2d, (x, y), counts_selection.ravel(),
            p0=[mux0, muy0, sigmax0, sigmay0, a0], bounds=([0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, np.inf, np.inf, np.inf]))

        # Check goodness of fit and update the goodness variable
        if popt[4] < a0/2:
            goodness -= 1
        if popt[0] < x_edges[0] or popt[0] > x_edges[-1]:
            goodness -= 1
        if popt[1] < y_edges[0] or popt[1] > y_edges[-1]:
            goodness -= 1
        if popt[2] > x_edges[-1] - x_edges[0]:
            goodness -= 1
        if popt[3] > y_edges[-1] - y_edges[0]:
            goodness -= 1

        # Calculate parameter errors
        perr = np.sqrt(np.diag(pcov))
        for j in range(len(perr)):
            if perr[j]/popt[j] > 0.5:
                goodness -= 1

        # Check if the fitting interval is too small in either dimension
        if minima[1] - minima[0] <= gap or minima[3] - minima[2] <= gap:
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

def plot_cumulative_figure(m_clean: np.ndarray, all_the_labels: np.ndarray,
    list_of_states: list[StateMulti], filename: str):
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

    if m_clean.shape[2] == 3:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        dir0 = [0, 0, 1]
        dir1 = [1, 2, 2]
        ax0 = [0, 0, 1]
        ax1 = [0, 1, 0]

        for k in range(3):
            d_0 = dir0[k]
            d_1 = dir1[k]
            a_0 = ax0[k]
            a_1 = ax1[k]
            # Plot the individual trajectories
            id_max, id_min = 0, 0
            for idx, mol in enumerate(m_clean):
                if np.max(mol) == np.max(m_clean):
                    id_max = idx
                if np.min(mol) == np.min(m_clean):
                    id_min = idx

            line_w = 0.05
            max_t = all_the_labels.shape[1]
            m_resized = m_clean[:, :max_t:, :]
            step = 5 if m_resized.size > 1000000 else 1

            for i, mol in enumerate(m_resized[::step]):
                ax[a_0][a_1].plot(mol.T[d_0], mol.T[d_1],
                    c='black', lw=line_w, rasterized=True, zorder=0)
                color_list = all_the_labels[i*step]
                ax[a_0][a_1].scatter(mol.T[d_0], mol.T[d_1], c=color_list,
                    cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

                color_list = all_the_labels[id_min]
                ax[a_0][a_1].plot(m_resized[id_min].T[d_0], m_resized[id_min].T[d_1],
                    c='black', lw=line_w, rasterized=True, zorder=0)
                ax[a_0][a_1].scatter(m_resized[id_min].T[d_0], m_resized[id_min].T[d_1],
                    c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5,
                    rasterized=True)
                color_list = all_the_labels[id_max]
                ax[a_0][a_1].plot(m_resized[id_max].T[d_0], m_resized[id_max].T[d_1],
                    c='black', lw=line_w, rasterized=True, zorder=0)
                ax[a_0][a_1].scatter(m_resized[id_max].T[d_0], m_resized[id_max].T[d_1],
                    c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5,
                    rasterized=True)

                # Plot the Gaussian distributions of states
                if k == 0:
                    for state in list_of_states:
                        ellipse = Ellipse(tuple(state.mean),
                            state.axis[d_0], state.axis[d_1], color='black', fill=False)
                        ax[a_0][a_1].add_patch(ellipse)

            # Set plot titles and axis labels
            ax[a_0][a_1].set_xlabel(r'Signal ' + str(d_0))
            ax[a_0][a_1].set_ylabel(r'Signal ' + str(d_1))

        ax[1][1].axis('off')
        fig.savefig('output_figures/' + filename + '.png', dpi=600)
        plt.close(fig)

        # ax: Axes3D = fig.add_subplot(111, projection='3d')

        # # Plot the individual trajectories
        # id_max, id_min = 0, 0
        # for idx, mol in enumerate(m_clean):
        #     if np.max(mol) == np.max(m_clean):
        #         id_max = idx
        #     if np.min(mol) == np.min(m_clean):
        #         id_min = idx

        # line_w = 0.05
        # max_t = all_the_labels.shape[1]
        # m_resized = m_clean[:, :max_t:, :]

        # step = 5 if m_resized.size > 1000000 else 1
        # for i, mol in enumerate(m_resized[::step]):
        #     ax.plot(mol.T[0], mol.T[1], mol.T[2], c='black', lw=line_w, rasterized=True, zorder=0)
        #     color_list = all_the_labels[i*step]
        #     ax.scatter(mol.T[0], mol.T[1], mol.T[2],
        #         c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)

        # color_list = all_the_labels[id_min]
        # ax.plot(m_resized[id_min].T[0], m_resized[id_min].T[1],
        #     m_resized[id_min].T[2], c='black', lw=line_w, rasterized=True, zorder=0)
        # ax.scatter(m_resized[id_min].T[0], m_resized[id_min].T[1], m_resized[id_min].T[2],
        #     c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)
        # color_list = all_the_labels[id_max]
        # ax.plot(m_resized[id_max].T[0], m_resized[id_max].T[1],
        #     m_resized[id_max].T[2], c='black', lw=line_w, rasterized=True, zorder=0)
        # ax.scatter(m_resized[id_max].T[0], m_resized[id_max].T[1], m_resized[id_max].T[2],
        #     c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, rasterized=True)

        # # Plot the Gaussian distributions of states
        # for s_id, state in enumerate(list_of_states):
        #     ang_u = np.linspace(0, 2*np.pi, 100)
        #     ang_v = np.linspace(0, np.pi, 100)
        #     point_x = state.axis[0]*np.outer(np.cos(ang_u), np.sin(ang_v)) + state.mean[0]
        #     point_y = state.axis[1]*np.outer(np.sin(ang_u), np.sin(ang_v)) + state.mean[1]
        #     point_z = state.axis[2]*np.outer(np.ones_like(ang_u), np.cos(ang_v)) + state.mean[2]
        #     ax.plot_surface(point_x, point_y, point_z, alpha=0.25, color=colors_from_cmap[s_id + 1])

        # # Set plot titles and axis labels
        # ax.set_xlabel(r'$x$')
        # ax.set_ylabel(r'$y$')
        # ax.set_zlabel(r'$z$')
    elif m_clean.shape[2] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        # ax = plt.axes()

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(m_clean):
            if np.max(mol) == np.max(m_clean):
                id_max = idx
            if np.min(mol) == np.min(m_clean):
                id_min = idx

        line_w = 0.05
        max_t = all_the_labels.shape[1]
        m_resized = m_clean[:, :max_t:, :]
        step = 5 if m_resized.size > 1000000 else 1

        for i, mol in enumerate(m_resized[::step]):
            ax.plot(mol.T[0], mol.T[1], c='black', lw=line_w, rasterized=True, zorder=0)
            color_list = all_the_labels[i*step]
            ax.scatter(mol.T[0], mol.T[1], c=color_list, cmap=colormap, vmin=0, vmax=n_states-1,
                s=0.5, rasterized=True)

        color_list = all_the_labels[id_min]
        ax.plot(m_resized[id_min].T[0], m_resized[id_min].T[1],
            c='black', lw=line_w, rasterized=True, zorder=0)
        ax.scatter(m_resized[id_min].T[0], m_resized[id_min].T[1],
            c=color_list, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
        color_list = all_the_labels[id_max]
        ax.plot(m_resized[id_max].T[0], m_resized[id_max].T[1],
            c='black', lw=line_w, rasterized=True, zorder=0)
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

        fig.savefig('output_figures/' + filename + '.png', dpi=600)
        plt.close(fig)

