from functions import *

output_file = 'states_output.txt'
colormap = 'viridis'
show_plot = True

def all_the_input_stuff():
	# Read input parameters from files.
	data_directory, PAR = read_input_parameters()

	tmp_M = []
	for d in range(len(data_directory)):
		# Read raw data from the specified directory/files.
		M_raw = read_data(data_directory[d])

		# Remove initial frames based on 'tau_delay'.
		M_raw = M_raw[:, PAR[1]:]

		# Apply filtering on the data
		m = moving_average(M_raw, PAR[0])

		# Normalize the data to the range [0, 1].
		sig_max = np.max(m)
		sig_min = np.min(m)
		m = (m - sig_min)/(sig_max - sig_min)

		tmp_M.append(m)

	# Get the number of particles and total frames in the trajectory.
	for d in range(len(tmp_M) - 1):
		if tmp_M[d].shape != tmp_M[d + 1].shape :
			print('ERROR: The signals do not correspond. Abort.')
			return 

	tmp_M = np.array(tmp_M)
	M = np.transpose(tmp_M, axes=(1, 2, 0))

	# M = np.array([ [ [M0[n][t], M1[n][t]] for t in range(M0.shape[1]) ] for n in range(M0.shape[0]) ])
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

	Bins = []
	Counts = []
	for d in range(M.shape[2]):
		# Flatten the M matrix and compute histogram counts and bins
		flat_M = M[:,:,d].flatten()
		bins = 'auto' if len(PAR) < 7 else PAR[6]
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
		ax1.get_shared_x_axes().join(ax1, ax2)
		ax3.get_shared_y_axes().join(ax3, ax2)
		ax1.set_xticklabels([])
		ax3.set_yticklabels([])
		
		# Plot histograms
		ax1.stairs(Counts[0], Bins[0], fill=True)
		ax3.stairs(Counts[1], Bins[1], fill=True, orientation='horizontal')

		# Plot the individual trajectories in the first subplot (left side)
		step = 10 if M.size > 1000000 else 1
		for idx, mol in enumerate(M[::step]):
			ax2.plot(mol[:,0], mol[:,1], color='black', lw=0.1, alpha=0.5, rasterized=True)

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

def gauss_fit_max(M, bins, filename):
	print('* Gaussian fit...')
	number_of_sigmas = 2.0
	flat_M = M.reshape((M.shape[0]*M.shape[1], M.shape[2]))

	### 1. Histogram with 'auto' binning ###
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

			tmp_max = tuple([max_ind[i] for i in range(D) if i != dim])

			while min_id0 > 0 and data[tuple([min_id0] + list(tmp_max))] > data[tuple([min_id0 - 1] + list(tmp_max))]:
				min_id0 -= 1

			while min_id1 < data.shape[dim] - 1 and data[tuple([min_id1] + list(tmp_max))] > data[tuple([min_id1 + 1] + list(tmp_max))]:
				min_id1 += 1

			minima.extend([min_id0, min_id1])

		return minima

	minima = find_minima_around_max(counts, max_ind, gap)

	### 5. Try the fit between the minima and check its goodness ###
	popt_min = []
	for dim in range(M.shape[2]):
		flag_min, goodness_min, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap)
		popt[2] *= flat_M.T[0].size
		popt_min.extend(popt)

	### 6. Find the interval of half height ###
	def find_half_height_around_max(data, max_ind, gap):
		max_val = data.max()
		minima = []

		D = data.ndim
		for dim in range(D):
			half_id0 = max(max_ind[dim] - gap, 0)
			half_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

			tmp_max = tuple([max_ind[i] for i in range(D) if i != dim])

			while half_id0 > 0 and data[tuple([half_id0] + list(tmp_max))] > max_val/2:
				half_id0 -= 1

			while half_id1 < data.shape[dim] - 1 and data[tuple([half_id1] + list(tmp_max))] > max_val/2:
				half_id1 += 1

			minima.extend([half_id0, half_id1])

		return minima

	minima = find_half_height_around_max(counts, max_ind, gap)

	### 7. Try the fit between the minima and check its goodness ###
	popt_half = []
	for dim in range(M.shape[2]):
		flag_half, goodness_half, popt = custom_fit(dim, max_ind[dim], minima, edges[dim], counts, gap)
		popt[2] *= flat_M.T[0].size
		popt_half.extend(popt)

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

	### Find the tresholds for state identification
	C, a = [], []
	for dim in range(M.shape[2]):
		C.append(popt[3*dim])
		a.append(number_of_sigmas*popt[3*dim + 1])
	ellipse = [C, a]

	### Plot the distribution and the fitted Gaussians -- this clearly works only with 2-dimensional data
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
		ax.scatter(C[0], C[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse(C, a[0]/number_of_sigmas, a[1]/number_of_sigmas, color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse(C, a[0], a[1], color='r', fill=False)
		ax.add_patch(circle1)
		ax.add_patch(circle2)
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.0])
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
		ax[0][0].scatter(C[0], C[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([C[0], C[1]], a[0]/number_of_sigmas, a[1]/number_of_sigmas, color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([C[0], C[1]], a[0], a[1], color='r', fill=False)
		ax[0][0].add_patch(circle1)
		ax[0][0].add_patch(circle2)

		im = matplotlib.image.NonUniformImage(ax[0][1], interpolation='nearest')
		im.set_data(zcenters, ycenters, np.sum(counts, axis=1))
		ax[0][1].add_image(im)
		ax[0][1].scatter(C[2], C[1], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([C[2], C[1]], a[2]/number_of_sigmas, a[1]/number_of_sigmas, color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([C[2], C[1]], a[2], a[1], color='r', fill=False)
		ax[0][1].add_patch(circle1)
		ax[0][1].add_patch(circle2)

		im = matplotlib.image.NonUniformImage(ax[1][0], interpolation='nearest')
		im.set_data(xcenters, zcenters, np.sum(counts, axis=2))
		ax[1][0].add_image(im)
		ax[1][0].scatter(C[0], C[2], s=8.0, c='red')
		circle1 = matplotlib.patches.Ellipse([C[0], C[2]], a[0]/number_of_sigmas, a[2]/number_of_sigmas, color='r', fill=False)
		circle2 = matplotlib.patches.Ellipse([C[0], C[2]], a[0], a[2], color='r', fill=False)
		ax[1][0].add_patch(circle1)
		ax[1][0].add_patch(circle2)

		for a in ax:
			for b in a:
				b.set_xlim([0.0, 1.0])
				b.set_ylim([0.0, 1.0])

	# if show_plot:
	#  	plt.show()
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
				rescaled = shifted / np.array(ellipse[1])
				# rescaled = shifted / np.array([ellipse[1], ellipse[2]])
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
		bins=50
		if len(PAR) == 7:
			bins = PAR[6]
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
			list_of_states.pop()
			break
		else:
			M1 = M2

	all_the_labels, list_of_states = relabel_states_2D(all_the_labels, list_of_states)
	return all_the_labels, list_of_states

def plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing cumulative figure...')
	tau_window, tau_delay, t_conv, t_units = PAR[0], PAR[1], PAR[2], PAR[3]
	n_states = len(list_of_states) + 1

	# Create a color palette for plotting states
	palette = []
	cmap = cm.get_cmap(colormap, n_states)
	for i in range(cmap.N):
		rgba = cmap(i)
		palette.append(matplotlib.colors.rgb2hex(rgba))

	fig = plt.figure(figsize=(6, 6))
	ax = plt.axes(projection='3d')

	# Plot the individual trajectories --- if labels are for individual points
	step = 1 if M.size > 1000000 else 1
	max_T = all_the_labels.shape[1]
	for i, mol in enumerate(M[::step]):
		ax.plot(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c='black', lw=0.2, rasterized=True, zorder=0)
		ax.scatter(mol.T[0,:max_T], mol.T[1,:max_T], mol.T[2,:max_T], c=all_the_labels[i],
			cmap=cmap, vmin=0.0, vmax=np.max(np.unique(all_the_labels)), s=0.5, rasterized=True)

	# Plot the Gaussian distributions of states
	for S_id, S in enumerate(list_of_states):
		[mux, muy, muz] = S[1][0]
		[a, b, c] = S[1][1]
		u = np.linspace(0, 2*np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		x = a*np.outer(np.cos(u), np.sin(v)) + mux
		y = b*np.outer(np.sin(u), np.sin(v)) + muy
		z = c*np.outer(np.ones_like(u), np.cos(v)) + muz
		ax.plot_surface(x, y, z, alpha=0.25, color=palette[S_id])

	# Set plot titles and axis labels
	ax.set_xlabel('Signal 1')
	ax.set_ylabel('Signal 2')
	ax.set_zlabel('Signal 3')

	if show_plot:
		plt.show()
	fig.savefig('output_figures/' + filename + '.png', dpi=600)
	plt.close(fig)

def plot_paper_figure(M, PAR, all_the_labels, list_of_states):
	tau_window, tau_delay, t_conv, t_units = PAR[0], PAR[1], PAR[2], PAR[3]
	fig, ax = plt.subplots(1, 2, figsize=(9, 5))

	step = 10 if M.size > 1000000 else 1
	time = np.linspace(tau_delay*t_conv, (tau_delay + M.shape[1])*t_conv, M.shape[1])
	t_start = 0
	t_stop = t_start + 1500
	for idx, mol in enumerate(M[::step]):
		if idx > 0: continue	### For clarity, I'm showing only the signals related to the particle 0
		ax[0].plot(time[t_start:t_stop], mol[:,0][t_start:t_stop], lw=1, color='blue')
		ax[0].plot(time[t_start:t_stop], mol[:,1][t_start:t_stop], lw=1, color='orange')
	alpha = 0.2
	ax[0].axvspan(0, 250, alpha=alpha, facecolor='green')
	ax[0].axvspan(500, 750, alpha=alpha, facecolor='red')
	ax[0].axvspan(1300, 1450, alpha=alpha, facecolor='red')
	ax[0].axvspan(350, 400, alpha=alpha, facecolor='green')
	ax[0].set_ylim([0.0, 1.0])
	ax[0].set_xlabel(r'Simulation time $t$')
	ax[0].set_ylabel(r'Signals')

	n_states = len(list_of_states) + 1
	cmap = cm.get_cmap(colormap, n_states)
	step = 10 if M.size > 1000000 else 1

	### If labels are for the single points
	max_T = all_the_labels.shape[1]
	for i, mol in enumerate(M[::step]):
		ax[1].plot(mol.T[0,:max_T], mol.T[1,:max_T], c='black', lw=0.1, alpha=0.5, rasterized=True, zorder=0)
	for i, mol in enumerate(M[::step]):
		ax[1].scatter(mol.T[0,:max_T], mol.T[1,:max_T], c=all_the_labels[i],
			cmap=cmap, vmin=0.0, vmax=np.max(np.unique(all_the_labels)), s=0.5, rasterized=True)

	# Plot the Gaussian distributions of states on the right subplot (ax[1])
	for S_id, S in enumerate(list_of_states):
		circle1 = matplotlib.patches.Ellipse(S[1][0], S[1][1][0]/2.0, S[1][1][1]/2.0, color='red', fill=False)
		circle2 = matplotlib.patches.Ellipse(S[1][0], S[1][1][0], S[1][1][1], color='red', fill=False, linestyle='--')
		ax[1].add_patch(circle1)
		ax[1].add_patch(circle2)

	# Set plot titles and axis labels
	ax[1].set_xlabel('Signal 1')
	ax[1].set_ylabel('Signal 2')
	ax[1].set_xlim([0.0, 1.0])
	ax[1].set_ylim([0.0, 1.0])

	# letter_subplots(ax)
	plt.tight_layout()
	plt.show()
	fig.savefig('Fig3.png', dpi=600)

def main():
	M_raw, PAR, data_directory = all_the_input_stuff()
	M, PAR, data_directory, all_the_labels, list_of_states = all_the_input_stuff()
	plot_input_data(M, PAR, 'Fig0')

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)
	all_the_labels = assign_final_states_to_single_frames_2D(M, all_the_labels, PAR[0], list_of_states)

	if len(list_of_states) == 0:
		print('* No possible classification was found. ')
		return

	plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, 'Fig2')
#	plot_paper_figure(M, PAR, all_the_labels, list_of_states)

if __name__ == "__main__":
	main()
