from functions import *

output_file = 'states_output.txt'
colormap = 'viridis'
# colormap = 'copper'
show_plot = False

def all_the_input_stuff():
	# Read input parameters from files.
	data_directory, PAR = read_input_parameters()

	# Read raw data from the specified directory/files.
	M0_raw = read_data(data_directory[0])
	M1_raw = read_data(data_directory[1])

	# Remove initial frames based on 'tau_delay'.
	M0_raw = M0_raw[:, PAR[1]:]
	M1_raw = M1_raw[:, PAR[1]:]

	# Apply filtering on the data
	M0 = moving_average(M0_raw, PAR[0])
	M1 = moving_average(M1_raw, PAR[0])

	# Normalize the data to the range [0, 1].
	sig_max = np.max(M0)
	sig_min = np.min(M0)
	M0 = (M0 - sig_min)/(sig_max - sig_min)
	sig_max = np.max(M1)
	sig_min = np.min(M1)
	M1 = (M1 - sig_min)/(sig_max - sig_min)

	# Get the number of particles and total frames in the trajectory.
	if M0.shape != M1.shape :
		print('ERROR: The two signals do not correspond. Abort.')
		return 

	M = np.array([ [ [M0[n][t], M1[n][t]] for t in range(M0.shape[1]) ] for n in range(M0.shape[0]) ])
	total_particles = M.shape[0]
	total_time = M.shape[1]

	# Calculate the number of windows for the analysis.
	num_windows = int(total_time / PAR[0])

	# Print informative messages about trajectory details.
	print('\tTrajectory has ' + str(total_particles) + ' particles. ')
	print('\tTrajectory of length ' + str(total_time) + ' frames (' + str(total_time*PAR[2]) + ' ns). ')
	print('\tUsing ' + str(num_windows) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[2]) + ' ns). ')

	# Initialize an array to store labels for each window.
	all_the_labels = np.zeros((total_particles, num_windows))
	 # Initialize an empty list to store unique states in each window.
	list_of_states = []

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

	# Return required data for further analysis.
	return M, PAR, data_directory, all_the_labels, list_of_states

def plot_input_data(M, PAR, filename):
	tau_window, tau_delay, t_conv = PAR[0], PAR[1], PAR[2]

	# Flatten the M matrix and compute histogram counts and bins
	flat_M0 = M[:,:,0].flatten()
	bins = 'auto' if len(PAR) < 6 else PAR[5]
	counts0, bins0 = np.histogram(flat_M0, bins=bins, density=True)
	counts0 *= flat_M0.size

	flat_M1 = M[:,:,1].flatten()
	bins = 'auto' if len(PAR) < 6 else PAR[5]
	counts1, bins1 = np.histogram(flat_M0, bins=bins, density=True)
	counts1 *= flat_M1.size

	# Create a plot with two subplots (side-by-side)
	fig = plt.figure(figsize=(9, 9))
	grid = fig.add_gridspec(4, 4)
	ax1 = fig.add_subplot(grid[0:1, 0:3])
	ax2 = fig.add_subplot(grid[1:4, 0:3])
	ax3 = fig.add_subplot(grid[1:4, 3:4])
	ax1.get_shared_x_axes().join(ax1, ax2)
	ax3.get_shared_y_axes().join(ax3, ax2)
	ax1.set_xticklabels([])
	ax3.set_yticklabels([])
	
	# Plot histograms
	ax1.stairs(counts0, bins0, fill=True)
	ax3.stairs(counts1, bins1, fill=True, orientation='horizontal')

	# Plot the individual trajectories in the first subplot (left side)
	step = 10 if M.size > 1000000 else 1
	for idx, mol in enumerate(M[::step]):
		ax2.plot(mol[:,0], mol[:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)

	# Set labels and titles for the plots
	ax2.set_ylabel('Signal 1')
	ax2.set_xlabel('Signal 2')

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def gauss_fit_max(M, bins, filename):
	print('* Gaussian fit...')
	number_of_sigmas = 2.0
	flat_M = np.reshape(M, (M.shape[0]*M.shape[1], 2), order='F')

	### 1. Histogram with 'auto' binning ###
	counts, xedges, yedges = np.histogram2d(flat_M.T[0], flat_M.T[1], bins=50, density=True)
	gap = 1
	if xedges.size > 40 and yedges.size > 40:
		gap = 3

	### 2. Smoothing with tau = 3 ###
	counts = moving_average_2D(counts, gap)

	### 3. Find the maximum ###
	max_val = counts.max()
	for i, c1 in enumerate(counts):
		for j, c2 in enumerate(c1):
			if c2 == max_val:
				max_ind = [i, j]
				break

	### 4. Find the minima surrounding it ###
	### Along x ###
	min_idx0 = np.max([max_ind[0] - gap, 0])
	min_idx1 = np.min([max_ind[0] + gap, counts.shape[0] - 1])
	while min_idx0 > 0 and counts[min_idx0][max_ind[1]] > counts[min_idx0 - 1][max_ind[1]]:
		min_idx0 -= 1
	while min_idx1 < counts.shape[0] - 1 and counts[min_idx1][max_ind[1]] > counts[min_idx1 + 1][max_ind[1]]:
		min_idx1 += 1
	### Along y ###
	min_idy0 = np.max([max_ind[1] - gap, 0])
	min_idy1 = np.min([max_ind[1] + gap, counts.shape[1] - 1])
	while min_idy0 > 0 and counts[max_ind[0]][min_idy0] > counts[max_ind[0]][min_idy0 - 1]:
		min_idy0 -= 1
	while min_idy1 < counts.shape[1] - 1 and counts[max_ind[0]][min_idy1] > counts[max_ind[0]][min_idy1 + 1]:
		min_idy1 += 1
	minima = [min_idx0, min_idx1, min_idy0, min_idy1]

	### 5. Try the fit between the minima and check its goodness ###
	flag_min, goodness_min, popt_min = fit_2D(max_ind, minima, xedges, yedges, counts, gap)
	popt_min[4] *= flat_M.T[0].size

	### 6. Find the inrterval of half height ###
	### Along x
	half_idx0 = np.max([max_ind[0] - gap, 0])
	half_idx1 = np.min([max_ind[0] + gap, counts.shape[0] - 1])
	while half_idx0 > 0 and counts[half_idx0][[max_ind[1]]] > max_val/2:
		half_idx0 -= 1
	while half_idx1 < counts.shape[0] - 1 and counts[half_idx1][max_ind[1]] > max_val/2:
		half_idx1 += 1
	## Along y
	half_idy0 = np.max([max_ind[1] - gap, 0])
	half_idy1 = np.min([max_ind[1] + gap, counts.shape[1] - 1])
	while half_idy0 > 0 and counts[max_ind[0]][half_idy0] > max_val/2:
		half_idy0 -= 1
	while half_idy1 < counts.shape[1] - 1 and counts[max_ind[0]][half_idy1] > max_val/2:
		half_idy1 += 1
	minima = [half_idx0, half_idx1, half_idy0, half_idy1]

	### 7. Try the fit between the minima and check its goodness ###
	flag_half, goodness_half, popt_half = fit_2D(max_ind, minima, xedges, yedges, counts, gap)
	popt_half[4] *= flat_M.T[0].size

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
		return [], []

	with open(output_file, 'a') as f:
		print('\n', file=f)
		print(f'\tmu = [{popt[0]:.4f}, {popt[1]:.4f}], sigma = [{popt[2]:.4f}, {popt[3]:.4f}], area = {popt[4]:.4f}')
		print(f'\tmu = [{popt[0]:.4f}, {popt[1]:.4f}], sigma = [{popt[2]:.4f}, {popt[3]:.4f}], area = {popt[4]:.4f}', file=f)
		print('\tFit goodness = ' + str(goodness), file=f)

	### Find the tresholds for state identification
	C = np.array([popt[0], popt[1]])
	a = number_of_sigmas*popt[2]
	b = number_of_sigmas*popt[3]
	ellipse = [C, a, b]

	### Plot the distribution and the fitted Gaussians
	fig, ax = plt.subplots(figsize=(6, 6))
	im = matplotlib.image.NonUniformImage(ax, interpolation='nearest')
	xcenters = (xedges[:-1] + xedges[1:]) / 2
	ycenters = (yedges[:-1] + yedges[1:]) / 2
	im.set_data(xcenters, ycenters, counts.T)
	ax.add_image(im)
	ax.scatter(popt[0], popt[1], s=8.0, c='red')
	circle1 = matplotlib.patches.Ellipse(C, popt[2], popt[3], color='r', fill=False)
	circle2 = matplotlib.patches.Ellipse(C, a, b, color='r', fill=False)
	ax.add_patch(circle1)
	ax.add_patch(circle2)
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.0])

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return popt, ellipse

def find_stable_trj(M, tau_window, ellipse, list_of_states, all_the_labels, offset):
	print('* Finding stable windows...')

	# Calculate the number of windows in the trajectory
	number_of_windows = int(M.shape[1]/tau_window)

	# Initialize an empty list to store non-stable windows
	M2 = []

	# Initialize a counter to keep track of the number of stable windows found
	counter = 0

	# Loop over each particle's trajectory
	for i, r in enumerate(M):
		# Loop over each window in the trajectory
		for w in range(number_of_windows):
			# Check if the window is already assigned to a state with a label > 0
			if all_the_labels[i][w] > 0.5:
				# If yes, skip this window and continue to the next one
				continue
			else:
				# If the window is not assigned to any state yet, extract the window's data
				r_w = r[w*tau_window:(w + 1)*tau_window]
				# Check if the window is stable (all data points within the specified ellispe)
				shifted = r_w - ellipse[0]
				rescaled = shifted / np.array([ellipse[1], ellipse[2]])
				squared_distances = np.sum(rescaled**2, axis=1)
				if np.max(squared_distances) <= 1.0:
					# If stable, assign the window to the current state offset and increment the counter
					all_the_labels[i][w] = offset + 1
					counter += 1
				else:
					# If not stable, add the window's data to the list of non-stable windows
					M2.append(r_w)

	# Calculate the fraction of stable windows found
	fw = counter/(all_the_labels.size)

	# Print the fraction of stable windows
	with open(output_file, 'a') as f:
		print(f'\tFraction of windows in state {offset} = {fw:.3}')
		print(f'\tFraction of windows in state {offset} = {fw:.3}', file=f)
	
	# Update the fraction of stable windows for the current state in the list_of_states
	list_of_states[-1][2] = fw

	# Convert the list of non-stable windows to a NumPy array
	M2 = np.array(M2)

	# Calculate the fraction of stable windows with respect to the total number of windows
	overall_fw = counter / (len(M) * number_of_windows)

	# Return the array of non-stable windows, the fraction of stable windows, and the updated list_of_states
	return M2, overall_fw, list_of_states

def iterative_search(M, PAR, all_the_labels, list_of_states):
	M1 = M
	iteration_id = 1
	states_counter = 0
	while True:
		### Locate and fit maximum in the signal distribution
		bins='auto'
		if len(PAR) == 6:
			bins=PAR[5]
		popt, ellipse = gauss_fit_max(M1, bins, 'output_figures/Fig1_' + str(iteration_id))
		if len(popt) == 0:
			break

		list_of_states.append([popt, ellipse, 0.0])

		### Find the windows in which the trajectories are stable in the maximum
		M2, c, list_of_states = find_stable_trj(M, PAR[0], ellipse, list_of_states, all_the_labels, states_counter)

		states_counter += 1
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c <= 0.0 or M2.size == 0:
			break
		else:
			M1 = M2

	all_the_labels, list_of_states = relabel_states_2D(all_the_labels, list_of_states)
	return all_the_labels, list_of_states

def plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing cumulative figure...')
	tau_window, tau_delay, t_conv, t_units = PAR[0], PAR[1], PAR[2], PAR[3]
	n_states = np.unique(all_the_labels).size

	# Create a color palette for plotting states
	palette = []
	cmap = cm.get_cmap(colormap, n_states)
	for i in range(cmap.N):
		rgba = cmap(i)
		palette.append(matplotlib.colors.rgb2hex(rgba))

	fig, ax = plt.subplots(figsize=(6, 6))

	# Plot the individual trajectories on the left subplot (ax[0])
	step = 10 if M.size > 1000000 else 1
	max_T = all_the_labels.shape[1]*tau_window
	for i, mol in enumerate(M[::step]):
		ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], c='black', lw=0.1, alpha=0.5, rasterized=True, zorder=0)
	for i, mol in enumerate(M[::step]):
		colors = np.repeat(all_the_labels[i], tau_window)
		ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], c=colors,
			cmap=colormap, vmin=0.0, vmax=np.max(np.unique(all_the_labels)), s=0.5, rasterized=True)

	# Plot the Gaussian distributions of states on the right subplot (ax[1])
	for S in list_of_states:
		circle1 = matplotlib.patches.Ellipse(S[1][0], S[0][2], S[0][3], color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse(S[1][0], S[1][1], S[1][2], color='r', fill=False)
		ax.add_patch(circle1)
		ax.add_patch(circle2)

	# Set plot titles and axis labels
	ax.set_xlabel('Signal 1')
	ax.set_ylabel('Signal 2')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.0])

	plt.show()
	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def plot_one_trajectory(M, PAR, all_the_labels, filename):
	tau_window, tau_delay, t_conv, t_units, example_ID = PAR[0], PAR[1], PAR[2], PAR[3], PAR[4]

	# Get the signal of the example particle
	signal = M[example_ID]

	# Create time values for the x-axis
	times = np.arange(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1]) * t_conv

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

def plot_all_trajectory_with_histos(M, PAR, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]

	fig = plt.figure()
	ax0 = plt.subplot(2, 4, 1)
	ax1 = plt.subplot(2, 4, 2)
	ax2 = plt.subplot(2, 4, 3)
	ax3 = plt.subplot(2, 4, 4)
	ax4 = plt.subplot(2, 1, 2)
	axes = [ax0, ax1, ax2, ax3, ax4]

	t_lim = np.array([tau_delay + int(tau_window/2), (tau_delay + int(tau_window/2) + M.shape[1])])*t_conv
	time = np.linspace(t_lim[0], t_lim[1], M.shape[1])

	if M.shape[1] > 1000:
		for mol in M[::10]:
			ax4.plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax4.plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	block_t = int(M.shape[1]/4)
	for i in range(4):
		part_signal = M[:, :(i + 1)*block_t].flatten()
		counts, bins = np.histogram(part_signal, bins='auto', density=True)
		axes[i].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)
		if i > 0:
			axes[i].set_yticklabels([])

	fig.suptitle('Example particle: ID = ' + str(PAR[4]))
	ax4.set_xlabel('Time ' + PAR[3])
	ax4.set_ylabel('Normalized signal')
	ax4.set_xlim(t_lim)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
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

def transition_matrix(Delta, PAR, all_the_labels, filename):
	t_conv = PAR[2]
	print('* Computing transition matrix...')
	# Get the number of unique states in the all_the_labels array.
	n_states = np.unique(all_the_labels).size
	# Initialize an empty matrix 'T' to store the transition counts.
	T = np.zeros((n_states, n_states))

	# Loop over each molecule trajectory in all_the_labels.
	for mol in all_the_labels:
		# Loop through the time steps of the trajectory.
		for t in range(mol.size - Delta):
			# Get the current state 'id0' and the state after 'Delta' steps 'id1'.
			id0 = int(mol[t])
			id1 = int(mol[t + Delta])
			# Increment the corresponding transition count in the matrix 'T'.
			T[id0][id1] += 1.0

	# Normalize the transition matrix 'T' to obtain transition probabilities 'N'.
	N = T / T.sum(axis=1, keepdims=True)

	# P, D = linalg.eig(N)
	# print(D)
	# print(-Delta*t_conv/np.log(P))

	# Create a plot to visualize the transition probabilities.
	fig, ax = plt.subplots(figsize=(10, 8))
	# Display the transition probabilities as an image with a logarithmic color scale.
	im = ax.imshow(N, cmap='plasma', norm=LogNorm(vmin=N[N > 0].min(), vmax=N.max()))
	fig.colorbar(im, ax=ax, format='%.5f')

	# Add text labels for each transition probability on the plot.
	for (i, j), val in np.ndenumerate(N):
		ax.text(j, i, "{:.2f}".format(val), ha='center', va='center')

	# Set titles and axis labels for the plot.
	fig.suptitle(r'Transition probabilities, $\Delta t=$' + str(Delta*t_conv) + ' ' + PAR[3])
	ax.set_xlabel('To...')
	ax.set_ylabel('From...')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_xticks(np.arange(n_states))
	ax.set_yticks(np.arange(n_states))
	ax.set_xticklabels(range(n_states))
	ax.set_yticklabels(range(n_states))
	plt.tight_layout()

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def tau_sigma(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	data = []
	labels = []
	wc = 0
	for i, x in enumerate(M):
		current_label = all_the_labels[i][0]
		x_w = x[0:tau_window]
		for w in range(1, number_of_windows):
			 if all_the_labels[i][w] == current_label:
			 	x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
			 else:
			 	### Lag-1 autocorrelation for colored noise
				### fitting with x_n = \alpha*x_{n-1} + z_n, z_n Gaussian white noise
			 	x_smean = x_w - np.mean(x_w)
			 	alpha = 0.0
			 	var = 1.0
			 	try:
			 		alpha, var, _ = wavelet.ar1(x_smean)
			 		data.append([alpha, np.sqrt(var)])
			 		labels.append(current_label)
			 	except Warning:
			 		wc += 1
			 	x_w = x[tau_window*w:tau_window*(w + 1)]
			 	current_label = all_the_labels[i][w]

	print('\t-- ' + str(wc) + ' warnings generated --')
	data = np.array(data).T
	tau_c = 1/(1 - data[0])

	figa, axa = plt.subplots(figsize=(7.5, 4.8))
	axa.scatter(tau_c, data[1], c='xkcd:black', s=1.0)
	axa.set_xlabel(r'Correlation time $\tau_c$ [ps]')
	axa.set_ylabel(r'Gaussian noise amplitude $\sigma_n$')
	figa.savefig(filename + 'a.png', dpi=600)

	figb, axb = plt.subplots(figsize=(7.5, 4.8))
	axb.scatter(tau_c, data[1], c=labels, s=1.0)
	axb.set_xlabel(r'Correlation time $\tau_c$ [ps]')
	axb.set_ylabel(r'Gaussian noise amplitude $\sigma_n$')
	# axb.legend()
	figb.savefig(filename + 'b.png', dpi=600)
	
	if show_plot:
		plt.show()

def state_statistics(M, PAR, all_the_labels, list_of_states, filename):
	print('* Computing some statistics on the enviroinments...')
	tau_window, t_conv, t_units = PAR[0], PAR[2], PAR[3]

	data = []
	labels = []
	# Iterate through the molecules in M
	for i, x in enumerate(M):
		mol_data = []
		mol_data_full = []
		mol_labels = []
		current_label = all_the_labels[i][0]  # Get the initial label of the molecule
		t0 = 0

		# Find the indices where the labels change
		label_changes = np.where(all_the_labels[i][1:] != all_the_labels[i][:-1])[0] + 1

		# Add the index of the last frame to label_changes
		label_changes = np.append(label_changes, len(all_the_labels[i]))

		# Iterate through the label changes and calculate statistics for each environment
		for t in label_changes:
			x_t = M[i][t0:t]  # Get the complete extent of the environment
			mol_data.append([x_t.size * t_conv, np.mean(x_t)])  # Store statistics for the environment
			mol_data_full.append(x_t)
			mol_labels.append(int(current_label))  # Store the label for the environment

			if t < len(M[i]):
				current_label = all_the_labels[i][t]  # Update the current label

			t0 = t

		### Here I want to remove the transitions which are too small (Chi2 distribution treshold) ###
		tmp_j_to_del = []
		for j in range(len(mol_data) - 1):
			Delta = mol_data[j + 1][1] - mol_data[j][1]
			sigma1 = list_of_states[mol_labels[j]][0][1]
			sigma2 = list_of_states[mol_labels[j + 1]][0][1]
			Chi2 = Delta**2/(sigma1**2 + sigma2**2)
			if Chi2 < 1.5:
				if mol_data[j][0] <= mol_data[j + 1][0]:
					tmp_j_to_del.append(j)
				else:
					tmp_j_to_del.append(j + 1)
		tmp_j_to_del.append(len(mol_data) - 1)

		mol_data = [mol_data[j] for j in range(len(mol_data)) if j not in tmp_j_to_del]
		mol_labels = [mol_labels[j] for j in range(len(mol_labels)) if j not in tmp_j_to_del]

		if len(data) == 0:
			data = mol_data
			labels = mol_labels
		elif len(mol_data) > 0:
			data = np.concatenate((data, mol_data))
			labels = np.concatenate((labels, mol_labels))

	data = np.array(data)

	# Characterization of the states
	state_points = []
	for s in np.unique(labels):
		ID_s = np.where(labels == s)[0]
		state_data = data[ID_s]
		state_points.append([np.mean(state_data[:, 0]), np.mean(state_data[:, 1]), np.std(state_data[:, 0]), np.std(state_data[:, 1])])
	state_points_tr = np.array(state_points).T

	# Append environment statistics to the output file
	with open(output_file, 'a') as f:
		print('\nEnviroinments\n', file=f)
		for E in state_points:
			print(E, file=f)

	# Create a scatter plot of the data points
	fig, ax = plt.subplots()

	scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=1.0, cmap=colormap)
	# Add error bars representing standard deviations of T and A
	ax.errorbar(state_points_tr[0], state_points_tr[1], xerr=state_points_tr[2], yerr=state_points_tr[3],
		marker='o', ms=3.0, c='black', lw=0.0, elinewidth=1.0, capsize=2.5)

	# Set plot titles and labels
	fig.suptitle('Statistics of the dynamic enviroinments')
	ax.set_xlabel(r'Duration $T$ ' + t_units)
	ax.set_ylabel(r'Amplitude $A$')
	ax.legend(*scatter.legend_elements())

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)

def transition_statistics(M, PAR, all_the_labels, list_of_states, filename):
	print('* Computing some statistics on the transitions...')
	tau_window, t_conv, t_units = PAR[0], PAR[2], PAR[3]

	data = []
	labels = []
	for i, x in enumerate(M):
		mol_data = []
		mol_labels = []
		current_label = all_the_labels[i][0]  # Get the initial label of the molecule
		t0 = 0

		# Find the indices where the labels change
		label_changes = np.where(all_the_labels[i][1:] != all_the_labels[i][:-1])[0] + 1

		# Add the index of the last frame to label_changes
		label_changes = np.append(label_changes, len(all_the_labels[i]))

		# Iterate through the label changes and calculate statistics for each environment
		for t in label_changes:
			x_t = M[i][t0:t]  # Get the complete extent of the environment
			mol_data.append(x_t)  # Store the environment
			mol_labels.append(int(current_label))  # Store the label for the environment

			if t < len(M[i]):
				current_label = all_the_labels[i][t]  # Update the current label

			t0 = t

		data.append(mol_data)
		labels.append(mol_labels)

	n_states = np.unique(all_the_labels).size

	### Create dictionary
	dictionary_21 = np.empty((n_states, n_states))
	dictionary_12 = []
	c = 0
	for i in range(n_states):
		for j in range(n_states):
			if i != j:
				dictionary_21[i][j] = c
				c += 1
				dictionary_12.append(str(i) + '-->' + str(j))

	T = np.zeros((n_states, n_states))

	transition_data = []
	transition_labels = []
	for i in range(len(data)):
		for t in range(len(data[i]) - 1):
			state0 = labels[i][t]
			state1 = labels[i][t + 1]
			mu0 = list_of_states[state0][0][0]
			mu1 = list_of_states[state1][0][0]
			sigma0 = list_of_states[state0][0][1]
			sigma1 = list_of_states[state1][0][1]
			Chi2 = (np.mean(data[i][t]) - np.mean(data[i][t + 1]))**2 / (sigma0**2 + sigma1**2)
			if Chi2 > 1.0:
				T[state0][state1] += 1
				transition_data.append([data[i][t].size*t_conv, np.mean(data[i][t + 1]) - np.mean(data[i][t])])
				transition_labels.append(dictionary_21[state0][state1])

	# print(T)

	transition_data_tr = np.array(transition_data).T
	transition_labels = np.array(transition_labels)

	state_points = []
	for i in range(n_states):
		for j in range(n_states):
			s = dictionary_21[i][j]
			ID_s = np.where(transition_labels == s)[0]
			if ID_s.size == 0:
				continue
			T = np.mean(transition_data_tr[0][ID_s])
			sigma_T = np.std(transition_data_tr[0][ID_s])
			A = np.mean(transition_data_tr[1][ID_s])
			sigma_A = np.std(transition_data_tr[1][ID_s])
			color = dictionary_21[i][j]
			if i > j:
				color = dictionary_21[j][i]
			state_points.append([T, A, sigma_T, sigma_A, color])
	state_points_tr = np.array(state_points).T

	with open(output_file, 'a') as f:
		print('\nTransitions\n', file=f)
		for E in state_points:
			print(E, file=f)

	fig, ax = plt.subplots()
	scatter = ax.scatter(transition_data_tr[0], transition_data_tr[1], c=transition_labels, s=1.0, cmap='plasma', alpha=0.5)
	ax.errorbar(state_points_tr[0], state_points_tr[1], xerr=state_points_tr[2], yerr=state_points_tr[3], marker='o', ms=3.0, c='black', lw=0.0, elinewidth=1.0, capsize=2.5)
	ax.scatter(state_points_tr[0], state_points_tr[1], marker='s', s=30.0, c=state_points_tr[4], cmap='plasma')
	fig.suptitle('Transitions statistics')
	ax.set_xlabel(r'Waiting time $\Delta t$ ' + t_units)
	ax.set_ylabel(r'Transition amplitude $\Delta A$')
	handles, _ = scatter.legend_elements()
	tmp = []
	for fl in np.unique(transition_labels):
		tmp.append(dictionary_12[int(fl)])
	tmp = np.array(tmp)
	ax.legend(handles, tmp)
	# ax.set_xscale('log')
	fig.savefig('output_figures/' + filename + '.png', dpi=600)

	# fig1, ax1 = plt.subplots()
	# fig2, ax2 = plt.subplots()
	# for tr in [0.0, 2.0, 3.0, 5.0]:
	# 	tmp_data = transition_data_tr[0][transition_labels == tr]
	# 	counts, bins, _ = ax1.hist(tmp_data, bins='doane', density=True, histtype='step', label=dictionary_12[int(tr)])
	# 	counts, bins, _ = ax2.hist(tmp_data, bins='auto', density=True, histtype='step', cumulative=True, label=dictionary_12[int(tr)])

	# ax1.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	# ax1.set_ylabel(r'Probability density function PDF$(\Delta t)$')
	# ax1.set_yscale('log')
	# ax1.legend(loc='upper right')
	# fig1.savefig(filename + 'a.png', dpi=600)

	# ax2.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	# ax2.set_ylabel(r'Cumulative distribution function CDF$(\Delta t)$')
	# ax2.set_xscale('log')
	# ax2.legend(loc='lower right')
	# fig2.savefig(filename + 'b.png', dpi=600)

	if show_plot:
		plt.show()

def main():
	M, PAR, data_directory, all_the_labels, list_of_states = all_the_input_stuff()
	plot_input_data(M, PAR, 'Fig0')

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)
	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		return

	plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, 'Fig2')
	# # plot_all_trajectory_with_histos(M, PAR, 'Fig2a')
	# plot_one_trajectory(M, PAR, all_the_labels, 'Fig3')

	# print_mol_labels_fbf_gro(all_the_labels)
	# print_mol_labels_fbf_lam(all_the_labels)

	# for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 100, 200, 300])]):
	# 	sankey(all_the_labels, frame_list, 10, PAR[2], 'Fig4_' + str(i))
	# transition_matrix(PAR[0], PAR, all_the_labels, 'Fig4a')

	# state_statistics(M, PAR, all_the_labels, list_of_states, 'Fig5')
	# transition_statistics(M, PAR, all_the_labels, list_of_states, 'Fig5a')

if __name__ == "__main__":
	main()
