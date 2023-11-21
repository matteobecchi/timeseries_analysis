from classes import *
from functions import *

output_file = 'states_output.txt'
show_plot = False

def all_the_input_stuff():
	# Read input parameters from files.
	data_directory = read_input_data()
	PAR = Parameters('input_parameters.txt')

	tmp_M_raw = []
	for d in range(len(data_directory)):
		# Read raw data from the specified directory/files.
		M_raw = read_data(data_directory[d])

		# Remove initial frames based on 'tau_delay'.
		M_raw = M_raw[:, PAR.t_delay:]

		tmp_M_raw.append(M_raw)

	for d in range(len(tmp_M_raw) - 1):
		if tmp_M_raw[d].shape != tmp_M_raw[d + 1].shape :
			print('ERROR: The signals do not correspond. Abort.')
			return

	### Create files for output
	with open(output_file, 'w') as f:
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
	return tmp_M_raw, PAR

def preparing_the_data(tmp_M_raw, PAR):
	tau_window, t_smooth, t_conv, t_units = PAR.tau_w, PAR.t_smooth, PAR.t_conv, PAR.t_units

	M = []
	for d, M_raw in enumerate(tmp_M_raw):
		# Apply filtering on the data
		m = moving_average(M_raw, t_smooth)

		# Normalize the data to the range [0, 1].
		sig_max = np.max(m)
		sig_min = np.min(m)
		m = (m - sig_min)/(sig_max - sig_min)

		M.append(m)

	M = np.array(M)
	M_limits = [ [np.min(x), np.max(x) ] for x in M ]
	M = np.transpose(M, axes=(1, 2, 0))

	total_particles = M.shape[0]
	total_time = M.shape[1]
	# Calculate the number of windows for the analysis.
	num_windows = int(total_time / tau_window)

	# Print informative messages about trajectory details.
	print('\tTrajectory has ' + str(total_particles) + ' particles. ')
	print('\tTrajectory of length ' + str(total_time) + ' frames (' + str(total_time*t_conv) + ' ' + t_units + ').')
	print('\tUsing ' + str(num_windows) + ' windows of length ' + str(tau_window) + ' frames (' + str(tau_window*t_conv) + ' ' + t_units + ').')

	return M, M_limits

def plot_input_data(M, PAR, filename):
	tau_window, tau_delay, t_conv, t_units, bins = PAR.tau_w, PAR.t_delay, PAR.t_conv, PAR.t_units, PAR.bins
	Bins = []
	Counts = []
	for d in range(M.shape[2]):
		# Flatten the M matrix and compute histogram counts and bins
		flat_M = M[:,:,d].flatten()
		counts0, bins0 = np.histogram(flat_M, bins=bins, density=True)
		counts0 *= flat_M.size
		Bins.append(bins0)
		Counts.append(counts0)

	if M.shape[2] == 2:
		# Create a plot with two subplots (side-by-side)
		fig = plt.figure(figsize=(9, 9))
		grid = fig.add_gridspec(4, 4)
		ax1 = fig.add_subplot(grid[0:1, 0:3])
		ax2 = fig.add_subplot(grid[1:4, 0:3])
		ax3 = fig.add_subplot(grid[1:4, 3:4])
		# ax1.get_shared_x_axes().join(ax1, ax2)
		# ax3.get_shared_y_axes().join(ax3, ax2)
		ax1.set_xticklabels([])
		ax3.set_yticklabels([])
		
		# Plot histograms
		ax1.stairs(Counts[0], Bins[0], fill=True)
		ax3.stairs(Counts[1], Bins[1], fill=True, orientation='horizontal')

		# Plot the individual trajectories in the first subplot (left side)
		ID_max, ID_min = 0, 0
		for idx, mol in enumerate(M):
			if np.max(mol) == np.max(M):
				ID_max = idx
			if np.min(mol) == np.min(M):
				ID_min = idx
		step = 10 if M.size > 1000000 else 1
		for idx, mol in enumerate(M[::step]):
			ax2.plot(mol[:,0], mol[:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)
		ax2.plot(M[ID_min][:,0], M[ID_min][:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)
		ax2.plot(M[ID_max][:,0], M[ID_max][:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)

		# Set labels and titles for the plots
		ax2.set_ylabel('Signal 1')
		ax2.set_xlabel('Signal 2')

	elif M.shape[2] == 3:
		fig = plt.figure(figsize=(6, 6))
		ax = plt.axes(projection='3d')
		
		# Plot the individual trajectories
		step = 1 if M.size > 1000000 else 1
		for idx, mol in enumerate(M[::step]):
			ax.plot(mol[:,0], mol[:,1], mol[:,2], color='black', marker='o', ms=0.5, lw=0.2, alpha=1.0, rasterized=True)

		# Set labels and titles for the plots
		ax.set_xlabel('Signal 1')
		ax.set_ylabel('Signal 2')
		ax.set_zlabel('Signal 3')

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def gauss_fit_max(M, M_limits, bins, filename):
	print('* Gaussian fit...')
	flat_M = M.reshape((M.shape[0]*M.shape[1], M.shape[2]))

	### 1. Histogram with 'auto' binning ###
	if bins == 'auto':
		bins = max(int(np.power(M.size, 1/3)*2), 10)
	counts, edges = np.histogramdd(flat_M, bins=bins, density=True)
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

		D = data.ndim
		for dim in range(D):
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
	for dim in range(M.shape[2]):
		try:
			flag_min, goodness, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap, M_limits)
			popt[2] *= flat_M.T[0].size
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

		D = data.ndim
		for dim in range(D):
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
	for dim in range(M.shape[2]):
		try:
			flag_half, goodness, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap, M_limits)
			popt[2] *= flat_M.T[0].size
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

	if len(popt) != M.shape[2]*3:
		print('\tWARNING: this fit is not converging.')
		return None		

	### Find the tresholds for state identification
	mu, sigma, A, a = [], [], [], []
	for dim in range(M.shape[2]):
		mu.append(popt[3*dim])
		sigma.append(popt[3*dim + 1])
		A.append(popt[3*dim + 2])
	state = State_multi(np.array(mu), np.array(sigma), np.array(A))

	### Plot the distribution and the fitted Gaussians
	if M.shape[2] == 2:
		with open(output_file, 'a') as f:
			print('\n', file=f)
			print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}')
			print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}', file=f)
			print('\tFit goodness = ' + str(goodness), file=f)

		fig, ax = plt.subplots(figsize=(6, 6))
		im = matplotlib.image.NonUniformImage(ax, interpolation='nearest')
		xcenters = (edges[0][:-1] + edges[0][1:]) / 2
		ycenters = (edges[1][:-1] + edges[1][1:]) / 2
		im.set_data(xcenters, ycenters, counts.T)
		ax.add_image(im)
		ax.scatter(mu[0], mu[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse(mu, sigma[0], sigma[1], color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse(mu, state.a[0], state.a[1], color='r', fill=False)
		ax.add_patch(circle1)
		ax.add_patch(circle2)
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
	elif M.shape[2] == 3:
		with open(output_file, 'a') as f:
			print('\n', file=f)
			print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}')
			print(f'\tmu = [{popt[0]:.4f}, {popt[3]:.4f}, {popt[6]:.4f}], sigma = [{popt[1]:.4f}, {popt[4]:.4f}, {popt[7]:.4f}], area = {popt[2]:.4f}, {popt[5]:.4f}, {popt[8]:.4f}', file=f)
			print('\tFit goodness = ' + str(goodness), file=f)

		fig, ax = plt.subplots(2, 2, figsize=(6, 6))
		xcenters = (edges[0][:-1] + edges[0][1:]) / 2
		ycenters = (edges[1][:-1] + edges[1][1:]) / 2
		zcenters = (edges[2][:-1] + edges[2][1:]) / 2

		im = matplotlib.image.NonUniformImage(ax[0][0], interpolation='nearest')
		im.set_data(xcenters, ycenters, np.sum(counts, axis=0))
		ax[0][0].add_image(im)
		ax[0][0].scatter(mu[0], mu[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([mu[0], mu[1]], sigma[0], sigma[1], color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([mu[0], mu[1]], state.a[0], state.a[1], color='r', fill=False)
		ax[0][0].add_patch(circle1)
		ax[0][0].add_patch(circle2)

		im = matplotlib.image.NonUniformImage(ax[0][1], interpolation='nearest')
		im.set_data(zcenters, ycenters, np.sum(counts, axis=1))
		ax[0][1].add_image(im)
		ax[0][1].scatter(mu[2], mu[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([mu[2], mu[1]], sigma[2], sigma[1], color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([mu[2], mu[1]], state.a[2], state.a[1], color='r', fill=False)
		ax[0][1].add_patch(circle1)
		ax[0][1].add_patch(circle2)

		im = matplotlib.image.NonUniformImage(ax[1][0], interpolation='nearest')
		im.set_data(xcenters, zcenters, np.sum(counts, axis=2))
		ax[1][0].add_image(im)
		ax[1][0].scatter(mu[0], mu[2], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([mu[0], mu[2]], sigma[0], sigma[2], color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([mu[0], mu[2]], state.a[0], state.a[2], color='r', fill=False)
		ax[1][0].add_patch(circle1)
		ax[1][0].add_patch(circle2)

		for a in ax:
			for b in a:
				b.set_xlim(0.0, 1.0)
				b.set_ylim(0.0, 1.0)

	if show_plot:
	 	plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return state

def find_stable_trj(M, tau_window, state, all_the_labels, offset):
	print('* Finding stable windows...')

	# Calculate the number of windows in the trajectory
	number_of_windows = int(M.shape[1] / tau_window )

	### This version, done with for loops, whold be removed after the new one is tested #######################
	# # Initialize an empty list to store non-stable windows
	# M2 = []
	# # Initialize a counter to keep track of the number of stable windows found
	# counter = 0
	# # Loop over each particle's trajectory
	# for i, r in enumerate(M):
	# 	# Loop over each window in the trajectory
	# 	for w in range(number_of_windows):
	# 		if w == all_the_labels.shape[1]:
	# 			continue ## Why does this happen?
	# 		# Check if the window is already assigned to a state with a label > 0
	# 		if all_the_labels[i][w] > 0.5:
	# 			# If yes, skip this window and continue to the next one
	# 			continue
	# 		else:
	# 			# If the window is not assigned to any state yet, extract the window's data
	# 			r_w = r[w*tau_window:(w + 1)*tau_window]
	# 			# Check if the window is stable (all data points within the specified ellises)
	# 			shifted = r_w - state.mu
	# 			rescaled = shifted / state.a

	# 			squared_distances = np.sum(rescaled**2, axis=1)
	# 			if np.max(squared_distances) <= 1.0:
	# 				# If stable, assign the window to the current state offset and increment the counter
	# 				all_the_labels[i][w] = offset + 1
	# 				counter += 1
	# 			else:
	# 				# If not stable, add the window's data to the list of non-stable windows
	# 				M2.append(r_w)
	#############################################################################################################

	mask_unclassified = all_the_labels < 0.5
	M_reshaped = M[:, :number_of_windows*tau_window].reshape(M.shape[0], number_of_windows, tau_window, M.shape[2])
	shifted = M_reshaped - state.mu
	rescaled = shifted / state.a
	squared_distances = np.sum(rescaled**2, axis=3)
	mask_dist = np.max(squared_distances, axis=2) <= 1.0
	mask = mask_unclassified & mask_dist

	all_the_labels[mask] = offset + 1	# Label the stable windows in the new state
	counter = np.sum(mask)				# The number of stable windows found

	# Store non-stable windows in a list, for the next iteration
	M2 = []
	mask_remaining = mask_unclassified & ~mask
	for i, w in np.argwhere(mask_remaining):
		r_w = M[i, w*tau_window:(w + 1)*tau_window]
		M2.append(r_w)

	# Calculate the fraction of stable windows found
	fw = counter/(all_the_labels.size)

	# Print the fraction of stable windows
	with open(output_file, 'a') as f:
		print(f'\tFraction of windows in state {offset + 1} = {fw:.3}')
		print(f'\tFraction of windows in state {offset + 1} = {fw:.3}', file=f)
	
	# Convert the list of non-stable windows to a NumPy array
	M2 = np.array(M2)
	one_last_state = True
	if len(M2) == 0:
		one_last_state = False

	# Return the array of non-stable windows, the fraction of stable windows, and the updated list_of_states
	return M2, fw, one_last_state

def iterative_search(M, M_limits, PAR, name):
	tau_w, bins = PAR.tau_w, PAR.bins

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
		state = gauss_fit_max(M1, M_limits, bins, 'output_figures/' + name + 'Fig1_' + str(iteration_id))
		if state == None:
			print('Iterations interrupted because unable to fit a Gaussian over the histogram. ')			
			break

		### Find the windows in which the trajectories are stable in the maximum
		M2, c, one_last_state = find_stable_trj(M, tau_w, state, all_the_labels, states_counter)
		state.perc = c

		if c > 0.0:
			states_list.append(state)
		
		states_counter += 1
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c <= 0.0:
			print('Iterations interrupted because no data points have been assigned to the last state. ')
			break
		elif M2.size == 0:
			print('Iterations interrupted because all data points have been assigned to one state. ')
			break
		else:
			M1 = M2

	all_the_labels, list_of_states = relabel_states_2D(all_the_labels, states_list)
	return all_the_labels, list_of_states, one_last_state

def plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing cumulative figure...')
	n_states = len(list_of_states) + 1
	colormap = 'viridis'
	x = plt.get_cmap(colormap, n_states)
	colors_from_cmap = x(np.arange(0, 1, 1/n_states))
	colors_from_cmap[-1] = x(1.0)
	
	fig = plt.figure(figsize=(6, 6))
	if M.shape[2] == 3:
		ax = plt.axes(projection='3d')

		# Plot the individual trajectories
		ID_max, ID_min = 0, 0
		for idx, mol in enumerate(M):
			if np.max(mol) == np.max(M):
				ID_max = idx
			if np.min(mol) == np.min(M):
				ID_min = idx

		lw = 0.05

		step = 5 if M.size > 1000000 else 1
		max_T = all_the_labels.shape[1]
		for i, mol in enumerate(M[::step]):
			ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
			c = [ int(l) for l in all_the_labels[i*step] ]
			ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
		
		c = [ int(l) for l in all_the_labels[ID_min] ]
		ax.plot(M[ID_min].T[0,:max_T], M[ID_min].T[1,:max_T], M[ID_min].T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
		ax.scatter(M[ID_min].T[0,:max_T], M[ID_min].T[1,:max_T], M[ID_min].T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
		c = [ int(l) for l in all_the_labels[ID_max] ]
		ax.plot(M[ID_max].T[0,:max_T], M[ID_max].T[1,:max_T], M[ID_max].T[2,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
		ax.scatter(M[ID_max].T[0,:max_T], M[ID_max].T[1,:max_T], M[ID_max].T[2,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

		# Plot the Gaussian distributions of states
		for S_id, S in enumerate(list_of_states):
			u = np.linspace(0, 2*np.pi, 100)
			v = np.linspace(0, np.pi, 100)
			x = S.a[0]*np.outer(np.cos(u), np.sin(v)) + S.mu[0]
			y = S.a[1]*np.outer(np.sin(u), np.sin(v)) + S.mu[1]
			z = S.a[2]*np.outer(np.ones_like(u), np.cos(v)) + S.mu[2]
			ax.plot_surface(x, y, z, alpha=0.25, color=colors_from_cmap[S_id+1])

		# Set plot titles and axis labels
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
		ax.set_zlabel(r'$z$')
	elif M.shape[2] == 2:
		ax = plt.axes()

		# Plot the individual trajectories
		ID_max, ID_min = 0, 0
		for idx, mol in enumerate(M):
			if np.max(mol) == np.max(M):
				ID_max = idx
			if np.min(mol) == np.min(M):
				ID_min = idx

		lw = 0.05

		step = 5 if M.size > 1000000 else 1
		max_T = all_the_labels.shape[1]
		for i, mol in enumerate(M[::step]):
			ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
			c = [ int(l) for l in all_the_labels[i*step] ]
			ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

		c = [ int(l) for l in all_the_labels[ID_min] ]
		ax.plot(M[ID_min].T[0,:max_T], M[ID_min].T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
		ax.scatter(M[ID_min].T[0,:max_T], M[ID_min].T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)
		c = [ int(l) for l in all_the_labels[ID_max] ]
		ax.plot(M[ID_max].T[0,:max_T], M[ID_max].T[1,:max_T], c='black', lw=lw, rasterized=True, zorder=0)
		ax.scatter(M[ID_max].T[0,:max_T], M[ID_max].T[1,:max_T], c=c, cmap=colormap, vmin=0, vmax=n_states-1, s=0.5, rasterized=True)

		# Plot the Gaussian distributions of states
		for S_id, S in enumerate(list_of_states):
			ellipse = matplotlib.patches.Ellipse(S.mu, S.a[0], S.a[1], color='black', fill=False)
			ax.add_patch(ellipse)

		# Set plot titles and axis labels
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')

	plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def plot_one_trajectory(M, PAR, all_the_labels, filename):
	tau_window, tau_delay, t_conv, t_units, example_ID = PAR.tau_w, PAR.t_delay, PAR.t_conv, PAR.t_units, PAR.example_ID

	# Get the signal of the example particle
	signal_x = M[example_ID].T[0][:all_the_labels.shape[1]]
	signal_y = M[example_ID].T[1][:all_the_labels.shape[1]]

	fig, ax = plt.subplots(figsize=(6, 6))
	colormap = 'viridis'

	# Create a colormap to map colors to the labels of the example particle
	cmap = plt.get_cmap(colormap, int(np.max(np.unique(all_the_labels)) - np.min(np.unique(all_the_labels)) + 1))
	color = all_the_labels[example_ID]
	ax.plot(signal_x, signal_y, c='black', lw=0.1)

	ax.scatter(signal_x, signal_y, c=color, cmap=cmap, vmin=np.min(np.unique(all_the_labels)), vmax=np.max(np.unique(all_the_labels)), s=1.0)

	# Set plot titles and axis labels
	fig.suptitle('Example particle: ID = ' + str(example_ID))
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')

	plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def timeseries_analysis(M_raw, PAR):
	tau_w, t_smooth = PAR.tau_w, PAR.t_smooth
	name = str(t_smooth) + '_' + str(tau_w) + '_'
	M, M_limits = preparing_the_data(M_raw, PAR)
	plot_input_data(M, PAR, name + 'Fig0')

	all_the_labels, list_of_states, one_last_state = iterative_search(M, M_limits, PAR, name)
	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		# We need to free the memory otherwise it accumulates
		del M_raw
		del M
		del all_the_labels
		return 1, 1.0
	
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
	if M.shape[2] > 2:
		return

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
		# std_list.append(np.std(tmp, axis=0))

	# Plotting
	fig, ax = plt.subplots()
	for l, center in enumerate(center_list):
		x = center[:, 0]
		y = center[:, 1]
		ax.plot(x, y, label='ENV'+str(l), marker='o')
	fig.suptitle('Average time sequence inside each environments')
	ax.set_xlabel(r'Signal 1')
	ax.set_ylabel(r'Signal 2')
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.legend()

	plt.show()
	fig.savefig('output_figures/Fig4.png', dpi=600)

def full_output_analysis(M_raw, PAR):
	tau_w = PAR.tau_w
	M, M_limits = preparing_the_data(M_raw, PAR)
	plot_input_data(M, PAR, 'Fig0')

	all_the_labels, list_of_states, one_last_state = iterative_search(M, M_limits, PAR, '')
	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		return

	compute_cluster_mean_seq(M, all_the_labels, tau_w)
	all_the_labels = assign_single_frames(all_the_labels, tau_w)
	plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, 'Fig2')
	plot_one_trajectory(M, PAR, all_the_labels, 'Fig1')

	print_mol_labels_fbf_xyz(all_the_labels)
	print_signal_with_labels(M, all_the_labels)
	print_colored_trj_from_xyz('trajectory.xyz', all_the_labels, PAR)

def TRA_analysis(M_raw, PAR, perform_anew):
	### If you want to change the range of the parameters tested, this is the point ###
	t_smooth_max = 5 	# 5
	num_of_points = 20 	# 20
	Tau_window, T_smooth = param_grid(M_raw[0].shape[1], t_smooth_max, num_of_points)

	if perform_anew:
		### If the analysis hat to be performed anew ###
		number_of_states = []
		fraction_0 = []
		for tau_w in Tau_window:
			tmp = [tau_w]
			tmp1 = [tau_w]
			for t_s in T_smooth:
				print('\n* New analysis: ', tau_w, t_s)
				tmp_PAR = copy.deepcopy(PAR)
				tmp_PAR.tau_w = tau_w
				tmp_PAR.t_smooth = t_s
				n_s, f0 = timeseries_analysis(M_raw, tmp_PAR)
				tmp.append(n_s)
				tmp1.append(f0)
			number_of_states.append(tmp)
			fraction_0.append(tmp1)
		header = 'tau_window\t t_s = 1\t t_s = 2\t t_s = 3\t t_s = 4\t t_s = 5'
		np.savetxt('number_of_states.txt', number_of_states, delimiter=' ', header=header)
		np.savetxt('fraction_0.txt', fraction_0, delimiter=' ', header=header)
	else:
		### Otherwise, just do this ###
		number_of_states = np.loadtxt('number_of_states.txt')
		fraction_0 = np.loadtxt('fraction_0.txt')

	plot_TRA_figure(number_of_states, fraction_0, PAR)

def main():
	M_raw, PAR = all_the_input_stuff()
	TRA_analysis(M_raw, PAR, False)
	full_output_analysis(M_raw, PAR)

if __name__ == "__main__":
	main()
