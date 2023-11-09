from functions import *

output_file = 'states_output.txt'
colormap = 'viridis'
show_plot = True

def all_the_input_stuff():
	# Read input parameters from files.
	data_directory, PAR = read_input_parameters()

	# Read raw data from the specified directory/files.
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		print('\tERROR: data_directory.txt is missing or wrongly formatted. ')

	# Remove initial frames based on 'tau_delay'.
	M_raw = M_raw[:, PAR[2]:]

	### Create files for output
	with open(output_file, 'w') as f:
		f.write('# {0}, {1}, {2}'.format(*PAR))
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

	return M_raw, PAR

def preparing_the_data(M_raw, PAR):
	tau_window = PAR[0]
	t_smooth = PAR[1]

	# Apply filtering on the data
	M = moving_average(M_raw, t_smooth)
	# M = np.array([ butter_lowpass_filter(x, 1/t_smooth, 1, 2) for x in M_raw ])

	sig_max = np.max(M)
	sig_min = np.min(M)
	# Normalize the data to the range [0, 1]. Usually not needed. 
	# M = (M - sig_min)/(sig_max - sig_min)

	# Get the number of particles and total frames in the trajectory.
	total_particles = M.shape[0]
	total_time = M.shape[1]

	# Calculate the number of windows for the analysis.
	num_windows = int(total_time / tau_window)

	# Print informative messages about trajectory details.
	print('\tTrajectory has ' + str(total_particles) + ' particles. ')
	print('\tTrajectory of length ' + str(total_time) + ' frames (' + str(total_time*PAR[3]), str(PAR[4]) + ')')
	print('\tUsing ' + str(num_windows) + ' windows of length ' + str(tau_window) + ' frames (' + str(tau_window*PAR[3]), str(PAR[4]) + ')')

	return M, [sig_min, sig_max]

def plot_input_data(M, PAR, filename):
	# Extract relevant parameters from PAR
	tau_window, tau_delay, t_conv, t_units = PAR[0], PAR[2], PAR[3], PAR[4]

	# Flatten the M matrix and compute histogram counts and bins
	flat_M = M.flatten()
	bins = 'auto' if len(PAR) < 7 else PAR[6]
	counts, bins = np.histogram(flat_M, bins=bins, density=True)
	counts *= flat_M.size

	# Create a plot with two subplots (side-by-side)
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

	# Plot histogram in the second subplot (right side)
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal')

	# Compute the time array for the x-axis of the first subplot (left side)
	time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1], M.shape[1])*t_conv

	# Plot the individual trajectories in the first subplot (left side)
	step = 10 if M.size > 1000000 else 1
	for idx, mol in enumerate(M[::step]):
		ax[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5, rasterized=True)

	# Set labels and titles for the plots
	ax[0].set_ylabel('Signal')
	ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
	ax[1].set_xticklabels([])

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def gauss_fit_max(M, bins, filename):
	print('* Gaussian fit...')
	flat_M = M.flatten()

	### 1. Histogram with 'auto' binning ###
	counts, bins = np.histogram(flat_M, bins=bins, density=True)
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
	Bins = bins[min_id0:min_id1]
	Counts = counts[min_id0:min_id1]
	mu0 = bins[max_ind]
	sigma0 = (bins[min_id0] - bins[min_id1])/6
	A0 = counts[max_ind]*np.sqrt(np.pi)*sigma0
	try:
		popt_min, pcov = scipy.optimize.curve_fit(Gaussian, Bins, Counts, p0=[mu0, sigma0, A0])
		if popt_min[1] < 0:
			popt_min[1] = -popt_min[1]
			popt_min[2] = -popt_min[2]
		gauss_max = popt_min[2]*np.sqrt(np.pi)*popt_min[1]
		if gauss_max < A0/2:
			goodness_min -= 1
		popt_min[2] *= flat_M.size
		if popt_min[0] < Bins[0] or popt_min[0] > Bins[-1]:
			goodness_min -= 1
		if popt_min[1] > Bins[-1] - Bins[0]:
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
	Bins = bins[half_id0:half_id1]
	Counts = counts[half_id0:half_id1]
	mu0 = bins[max_ind]
	sigma0 = (bins[half_id0] - bins[half_id1])/6
	A0 = counts[max_ind]*np.sqrt(np.pi)*sigma0
	try:
		popt_half, pcov = scipy.optimize.curve_fit(Gaussian, Bins, Counts, p0=[mu0, sigma0, A0])
		if popt_half[1] < 0:
			popt_half[1] = -popt_half[1]
			popt_half[2] = -popt_half[2]
		gauss_max = popt_half[2]*np.sqrt(np.pi)*popt_half[1]
		if gauss_max < A0/2:
			goodness_half -= 1
		popt_half[2] *= flat_M.size
		if popt_half[0] < Bins[0] or popt_half[0] > Bins[-1]:
			goodness_half -= 1
		if popt_half[1] > Bins[-1] - Bins[0]:
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

	with open(output_file, 'a') as f:
		print('\n', file=f)
		print(f'\tmu = {state.mu:.4f}, sigma = {state.sigma:.4f}, area = {state.A:.4f}')
		print(f'\tmu = {state.mu:.4f}, sigma = {state.sigma:.4f}, area = {state.A:.4f}', file=f)
		print('\tFit goodness = ' + str(goodness), file=f)

	### Plot the distribution and the fitted Gaussians
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	fig, ax = plt.subplots()
	plot_histo(ax, counts, bins)
	ax.set_xlim(y_lim)
	tmp_popt = [state.mu, state.sigma, state.A/flat_M.size]
	ax.plot(np.linspace(bins[0], bins[-1], 1000), Gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return state

def find_stable_trj(M, tau_window, state, all_the_labels, offset):
	print('* Finding stable windows...')

	# Calculate the number of windows in the trajectory
	number_of_windows = all_the_labels.shape[1]

	# Initialize an empty list to store non-stable windows
	M2 = []

	# Initialize a counter to keep track of the number of stable windows found
	counter = 0

	# Loop over each particle's trajectory
	for i, x in enumerate(M):
		# Loop over each window in the trajectory
		for w in range(number_of_windows):
			# Check if the window is already assigned to a state with a label > 0
			if all_the_labels[i][w] > 0.5:
				# If yes, skip this window and continue to the next one
				continue
			else:
				# If the window is not assigned to any state yet, extract the window's data
				x_w = x[w*tau_window:(w + 1)*tau_window]
				# Check if the window is stable (all data points within the specified range)
				if np.amin(x_w) > state.th_inf[0] and np.amax(x_w) < state.th_sup[0]:
					# If stable, assign the window to the current state offset and increment the counter
					all_the_labels[i][w] = offset + 1
					counter += 1
				else:
					# If not stable, add the window's data to the list of non-stable windows
					M2.append(x_w)

	# Calculate the fraction of stable windows found
	fw = counter/(all_the_labels.size)

	# Print the fraction of stable windows
	with open(output_file, 'a') as f:
		print(f'\tFraction of windows in state {offset} = {fw:.3}')
		print(f'\tFraction of windows in state {offset} = {fw:.3}', file=f)
	
	# Convert the list of non-stable windows to a NumPy array
	M2 = np.array(M2)
	one_last_state = True
	if len(M2) == 0:
		one_last_state = False

	# Return the array of non-stable windows, the fraction of stable windows, and the updated list_of_states
	return M2, fw, one_last_state

def iterative_search(M, PAR, name):
	tau_w = PAR[0]
	
	# Initialize an array to store labels for each window.
	num_windows = int(M.shape[1] / tau_w)
	all_the_labels = np.zeros((M.shape[0], num_windows))

	states_list = []
	M1 = M
	iteration_id = 1
	states_counter = 0
	one_last_state = False
	while True:
		### Locate and fit maximum in the signal distribution
		bins='auto'
		if len(PAR) == 7:
			bins=PAR[6]
		# popt, th, state = gauss_fit_max(M1, bins, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
		state = gauss_fit_max(M1, bins, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
		if state == None:
			print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')
			break

		### Find the windows in which the trajectories are stable in the maximum
		M2, c, one_last_state = find_stable_trj(M, tau_w, state, all_the_labels, states_counter)
		state.perc = c

		states_list.append(state)
		states_counter += 1
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c <= 0.0:
			print('Iterations interrupted because no data point has been assigned to the last state. ')
			break
		else:
			M1 = M2

	atl, lis = relabel_states(all_the_labels, states_list)
	return atl, lis, one_last_state

def plot_cumulative_figure(M, PAR, list_of_states, filename):
	print('* Printing cumulative figure...')
	tau_window, tau_delay, t_conv, t_units = PAR[0], PAR[2], PAR[3], PAR[4]
	n_states = len(list_of_states)

	# Compute histogram of flattened M
	flat_M = M.flatten()
	bins = 'auto' if len(PAR) < 7 else PAR[6]
	counts, bins = np.histogram(flat_M, bins=bins, density=True)
	counts *= flat_M.size

	# Create a 1x2 subplots with shared y-axis
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

	# Plot the histogram on the right subplot (ax[1])
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

	# Create a color palette for plotting states
	palette = []
	cmap = cm.get_cmap(colormap, n_states)
	for i in range(cmap.N):
		rgba = cmap(i)
		palette.append(matplotlib.colors.rgb2hex(rgba))

	# Define time and y-axis limits for the left subplot (ax[0])
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1], M.shape[1])*t_conv

	# Plot the individual trajectories on the left subplot (ax[0])
	step = 10 if M.size > 1000000 else 1
	for idx, mol in enumerate(M[::step]):
		ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	# Plot the Gaussian distributions of states on the right subplot (ax[1])
	for S in range(n_states):
		popt = [list_of_states[S].mu, list_of_states[S].sigma, list_of_states[S].A]
		ax[1].plot(Gaussian(np.linspace(bins[0], bins[-1], 1000), *popt), np.linspace(bins[0], bins[-1], 1000), color=palette[S])

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

	plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def plot_one_trajectory(M, PAR, all_the_labels, filename):
	tau_window, tau_delay, t_conv, t_units, example_ID = PAR[0], PAR[2], PAR[3], PAR[4], PAR[5]

	# Get the signal of the example particle
	signal = M[example_ID][:all_the_labels.shape[1]]

	# Create time values for the x-axis
	times = np.arange(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1]) * t_conv
	times = times[:all_the_labels.shape[1]]

	# Create a figure and axes for the plot
	fig, ax = plt.subplots()

	# Create a colormap to map colors to the labels of the example particle
	cmap = plt.get_cmap(colormap, np.max(np.unique(all_the_labels)) - np.min(np.unique(all_the_labels)) + 1)
	color = all_the_labels[example_ID]
	ax.plot(times, signal, c='black', lw=0.1)

	# Plot the signal as a line and scatter plot with colors based on the labels
	ax.scatter(times, signal, c=color, cmap=cmap, vmin=np.min(np.unique(all_the_labels)), vmax=np.max(np.unique(all_the_labels)), s=1.0)

	# Add title and labels to the axes
	fig.suptitle('Example particle: ID = ' + str(example_ID))
	ax.set_xlabel('Time ' + t_units)
	ax.set_ylabel('Normalized signal')

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def sankey(all_the_labels, frame_list, aver_window, t_conv, filename):
	print('* Computing and plotting the averaged Sankey diagrams...')

	# Check if the required frame range is within the bounds of the input data.
	if frame_list[-1] + aver_window > all_the_labels.shape[1]:
		print('\tERROR: the required frame range is out of bound.')
		return

	# Determine the number of unique states in the data.
	n_states = np.unique(all_the_labels).size

	# Create arrays to store the source, target, and value data for the Sankey diagram.
	source = np.empty((frame_list.size - 1) * n_states**2)
	target = np.empty((frame_list.size - 1) * n_states**2)
	value = np.empty((frame_list.size - 1) * n_states**2)

	# Initialize a counter variable.
	c = 0

	# Create temporary lists to store node labels for the Sankey diagram.
	tmp_label1 = []
	tmp_label2 = []

	# Loop through the frame_list and calculate the transition matrix for each time window.
	for i, t0 in enumerate(frame_list[:-1]):
		# Calculate the time jump for the current time window.
		t_jump = frame_list[i + 1] - frame_list[i]

		# Initialize a matrix to store the transition counts between states.
		T = np.zeros((n_states, n_states))
	    
		# Iterate through the current time window and increment the transition counts in T.
		for t in range(t0, t0 + aver_window):
			for L in all_the_labels:
				T[int(L[t])][int(L[t + t_jump])] += 1

		# Store the source, target, and value for the Sankey diagram based on T.
		for n1 in range(len(T)):
			for n2 in range(len(T[n1])):
				source[c] = n1 + i * n_states
				target[c] = n2 + (i + 1) * n_states
				value[c] = T[n1][n2]
				c += 1

		# Calculate the starting and ending fractions for each state and store node labels.
		for n in range(n_states):
			starting_fraction = np.sum(T[n]) / np.sum(T)
			ending_fraction = np.sum(T.T[n]) / np.sum(T)
			if i == 0:
				tmp_label1.append('State ' + str(n) + ': ' + "{:.2f}".format(starting_fraction * 100) + '%')
			tmp_label2.append('State ' + str(n) + ': ' + "{:.2f}".format(ending_fraction * 100) + '%')

	# Concatenate the temporary labels to create the final node labels.
	label = np.concatenate((tmp_label1, np.array(tmp_label2).flatten()))

	# Generate a color palette for the Sankey diagram.
	palette = []
	cmap = cm.get_cmap(colormap, n_states)
	for i in range(cmap.N):
		rgba = cmap(i)
		palette.append(matplotlib.colors.rgb2hex(rgba))

	# Tile the color palette to match the number of frames.
	color = np.tile(palette, frame_list.size)

	# Create dictionaries to define the Sankey diagram nodes and links.
	node = dict(label=label, pad=30, thickness=20, color=color)
	link = dict(source=source, target=target, value=value)

	# Create the Sankey diagram using Plotly.
	Data = go.Sankey(link=link, node=node, arrangement="perpendicular")
	fig = go.Figure(Data)

	# Add the title with the time information.
	fig.update_layout(title='Frames: ' + str(frame_list * t_conv) + ' ns')

	if show_plot:
		fig.show()
	fig.write_image('output_figures/' + filename + '.png', scale=5.0)

def timeseries_analysis(M_raw, PAR):
	tau_w = PAR[0]
	t_smooth = PAR[1]
	name = str(t_smooth) + '_' + str(tau_w) + '_'
	M, M_range = preparing_the_data(M_raw, PAR)
	plot_input_data(M, PAR, name + 'Fig0')

	all_the_labels, list_of_states, one_last_state = iterative_search(M, PAR, name)
	
	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		# We need to free the memory otherwise it accumulates
		del M_raw
		del M
		del all_the_labels
		return 1, 1.0

	list_of_states, all_the_labels = set_final_states(list_of_states, all_the_labels, M_range)

	# We need to free the memory otherwise it accumulates
	del M_raw
	del M
	del all_the_labels

	fraction_0 = 1 - np.sum([ state.perc for state in list_of_states ])
	if one_last_state:
		return len(list_of_states) + 1, fraction_0
	else:
		return len(list_of_states), fraction_0

def compute_cluster_mean_seq(M, all_the_labels, tau_window):
	# Initialize lists to store cluster means and standard deviations
	center_list = []
	std_list = []

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
					tmp.append(M[i][t0:t1])

		# Calculate mean and standard deviation for the current cluster
		center_list.append(np.mean(tmp, axis=0))
		std_list.append(np.std(tmp, axis=0))

	# Plotting
	fig, ax = plt.subplots()
	x = range(tau_window)
	for l, center in enumerate(center_list):
		err_inf = center - std_list[l]
		err_sup = center + std_list[l]
		ax.fill_between(x, err_inf, err_sup, alpha=0.25)
		ax.plot(x, center, label='ENV'+str(l), marker='o')
	fig.suptitle('Average time sequence inside each environments')
	ax.set_xlabel(r'Time $t$ [frames]')
	ax.set_ylabel(r'Signal')
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.legend()

	plt.show()
	fig.savefig('output_figures/Fig4.png', dpi=600)

def full_output_analysis(M_raw, PAR):
	tau_w = PAR[0]
	t_smooth = PAR[1]
	M, M_range = preparing_the_data(M_raw, PAR)
	plot_input_data(M, PAR, 'Fig0')

	all_the_labels, list_of_states, one_last_state = iterative_search(M, PAR, '')
	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		return
	list_of_states, all_the_labels = set_final_states(list_of_states, all_the_labels, M_range)

	compute_cluster_mean_seq(M, all_the_labels, tau_w)

	all_the_labels = assign_single_frames(all_the_labels, tau_w)

	plot_cumulative_figure(M, PAR, list_of_states, 'Fig2')
	plot_one_trajectory(M, PAR, all_the_labels, 'Fig3')

	print_mol_labels_fbf_xyz(all_the_labels)

	# for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 100, 200])]):
	# 	sankey(all_the_labels, frame_list, 10, PAR[3], 'Fig4_' + str(i))

def TRA_analysis(M_raw, PAR):
	t_smooth_max = 5	# 5
	### The following is to have num_of_points log-spaced points
	num_of_points = 20	# 20
	base = (M_raw.shape[1] - t_smooth_max)**(1/num_of_points)
	tmp = [ int(base**n) + 1 for n in range(1, num_of_points + 1) ]
	tau_window = []
	[ tau_window.append(x) for x in tmp if x not in tau_window ]
	print('* Tau_w used:', tau_window)
	t_smooth = [ ts for ts in range(1, t_smooth_max + 1) ]
	print('* t_smooth used:', t_smooth)

	number_of_states = []
	fraction_0 = []
	for tau_w in tau_window:
		tmp = [tau_w]
		tmp1 = [tau_w]
		for t_s in t_smooth:
			print('\n* New analysis: ', tau_w, t_s)
			tmp_PAR = copy.deepcopy(PAR)
			tmp_PAR[0] = tau_w
			tmp_PAR[1] = t_s
			n_s, f0 = timeseries_analysis(M_raw, tmp_PAR)
			tmp.append(n_s)
			tmp1.append(f0)
		number_of_states.append(tmp)
		fraction_0.append(tmp1)

	np.savetxt('number_of_states.txt', number_of_states, delimiter=' ')
	np.savetxt('fraction_0.txt', fraction_0, delimiter=' ')

	### Otherwise, just do this ###
	# number_of_states = np.loadtxt('number_of_states.txt')[:, 1:]
	# fraction_0 = np.loadtxt('fraction_0.txt')[:, 1:]

	plot_TRA_figure(number_of_states, fraction_0, PAR)

def main():
	M_raw, PAR = all_the_input_stuff()
	# TRA_analysis(M_raw, PAR)
	full_output_analysis(M_raw, PAR)

if __name__ == "__main__":
	main()
