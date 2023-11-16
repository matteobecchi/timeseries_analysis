import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import copy
import math
import scipy.optimize
from scipy.signal import savgol_filter
from scipy.signal import butter,filtfilt
from matplotlib.colors import LogNorm
import plotly
plotly.__version__
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

def read_input_data():
	# Step 1: Attempt to read the content of 'data_directory.txt' file and load it into a NumPy array as strings.
	try:
		data_dir = np.loadtxt('data_directory.txt', dtype=str)
	except:
		print('\tdata_directory.txt file missing or wrongly formatted.')

	print('* Reading data from', data_dir)

	if data_dir.size == 1:
		return str(data_dir)
	else:
		return data_dir

def read_data(filename):
	# Check if the filename ends with a supported format.
	if filename.endswith(('.npz', '.npy', '.txt')):
		try:
			if filename.endswith('.npz'):
				with np.load(filename) as data:
					# Load the first variable (assumed to be the data) into a NumPy array.
					data_name = data.files[0]
					M = np.array(data[data_name])
			elif filename.endswith('.npy'):
				M = np.load(filename)
			else: # .txt file
				M = np.loadtxt(filename)
			print('\tOriginal data shape:', M.shape)
			return M
		except Exception as e:
			print(f'\tERROR: Failed to read data from {filename}. Reason: {e}')
			return None
	else:
		print('\tERROR: unsupported format for input file.')
		return None

def butter_lowpass_filter(x, cutoff, fs, order):
	nyq = 0.5
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = filtfilt(b, a, x)
	return y

def Savgol_filter(M, window):
	# Step 1: Set the polynomial order for the Savitzky-Golay filter.
	poly_order = 2

	# Step 2: Apply the Savitzky-Golay filter to each row (x) in the input data matrix 'M'.
	# The result is stored in a temporary array 'tmp'.
	# 'window' is the window size for the filter.
	tmp = np.array([savgol_filter(x, window, poly_order) for x in M])

	# Step 3: Since the Savitzky-Golay filter operates on a sliding window, 
	# it introduces edge artifacts at the beginning and end of each row.
	# To remove these artifacts, the temporary array 'tmp' is sliced to remove the unwanted edges.
	# The amount of removal on each side is half of the 'window' value, converted to an integer.
	return tmp[:, int(window/2):-int(window/2)]

def moving_average(data, window):
	# Step 1: Create a NumPy array 'weights' with the values 1.0 repeated 'window' times.
	# Then, divide each element of 'weights' by the 'window' value to get the average weights.
	weights = np.ones(window) / window

	# Step 2: Apply the moving average filter to the 'data' array using the 'weights' array.
	# The 'np.convolve' function performs a linear convolution between 'data' and 'weights'.
	# The result is a smoothed version of the 'data', where each point represents the weighted average of its neighbors.
	if data.ndim == 1:
		return np.convolve(data, weights, mode='valid')
	elif data.ndim >= 2:
		return np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'), axis=1, arr=data)
	else:
		raise ValueError('Invalid array dimension. Only 1D and 2D arrays are supported.')

def moving_average_2D(data, L):
	if L % 2 == 0:								# Check if L is an odd number
		raise ValueError("L must be an odd number.")
	half_width = (L - 1) // 2					# Calculate the half-width of the moving window
	result = np.zeros_like(data, dtype=float)	# Initialize the result array with zeros
	num_dims = data.ndim						# Get the number of dimensions in the input data

	for index in np.ndindex(data.shape):
		slices = tuple(slice(max(0, i - half_width), min(data.shape[dim], i + half_width + 1)) for dim, i in enumerate(index))
		subarray = data[slices]
		# Calculate the average if the subarray is not empty
		if subarray.size > 0:
			result[index] = subarray.mean()

	return result

def normalize_array(x):
	# Step 1: Calculate the mean value and the standard deviation of the input array 'x'.
	mean = np.mean(x)
	stddev = np.std(x)

	# Step 2: Create a temporary array 'tmp' containing the normalized version of 'x'.
	# To normalize, subtract the mean value from each element of 'x' and then divide by the standard deviation.
	# This centers the data around zero (mean) and scales it based on the standard deviation.
	tmp = (x - mean) / stddev

	# Step 3: Return the normalized array 'tmp', along with the calculated mean and standard deviation.
	# The returned values can be useful for further processing or to revert the normalization if needed.
	return tmp, mean, stddev

def plot_histo(ax, counts, bins):
	ax.stairs(counts, bins, fill=True)
	ax.set_xlabel(r'Normalized signal')
	ax.set_ylabel(r'Probability distribution')

def param_grid(total_time, t_smooth_max, n_windows):
	### The following is to have num_of_points log-spaced points
	base = (total_time - t_smooth_max)**(1/n_windows)
	tmp = [ int(base**n) + 1 for n in range(1, n_windows + 1) ]
	tau_window = []
	[ tau_window.append(x) for x in tmp if x not in tau_window ]
	print('* Tau_w used:', tau_window)

	t_smooth = [ ts for ts in range(1, t_smooth_max + 1) ]
	print('* t_smooth used:', t_smooth)

	return tau_window, t_smooth

def sigmoidal(x, A, B, alpha):
	return B + A/(1 + np.exp(x*alpha))

def Gaussian(x, m, sigma, A):
	# "m" is the Gaussians' mean value
	# "sigma" is the Gaussians' standard deviation
	# "A" is the Gaussian area
	return np.exp(-((x - m)/sigma)**2)*A/(np.sqrt(np.pi)*sigma)

def Gaussian_2D(r, mx, my, sigmax, sigmay, A):
	# "m" is the Gaussians' mean value (2d array)
	# "sigma" is the Gaussians' standard deviation matrix
	# "A" is the Gaussian area
	r[0] -= mx
	r[1] -= my
	arg = (r[0]/sigmax)**2 + (r[1]/sigmay)**2
	norm = np.pi*sigmax*sigmay
	gauss = np.exp(-arg)*A/norm
	return gauss.ravel()

def Gaussian_2D_full(r, mx, my, sigmax, sigmay, sigmaxy, A):
	# "m" is the Gaussians' mean value (2d array)
	# "sigma" is the Gaussians' standard deviation matrix
	# "A" is the Gaussian area
	r[0] -= mx
	r[1] -= my
	arg = (r[0]/sigmax)**2 + (r[1]/sigmay)**2 + 2*r[0]*r[1]/sigmaxy**2
	norm = np.pi*sigmax*sigmay/np.sqrt(1 - (sigmax*sigmay/sigmaxy**2)**2)
	gauss = np.exp(-arg)*A/norm
	return gauss.ravel()

def custom_fit(dim, max_ind, minima, edges, counts, gap):
	# Initialize flag and goodness variables
	flag = 1
	goodness = 5

	# Extract relevant data within the specified minima
	Edges = edges[minima[2*dim]:minima[2*dim + 1]]
	all_axes = tuple(i for i in range(counts.ndim) if i != dim)
	Counts = np.sum(counts, axis=all_axes)
	Counts = Counts[minima[2*dim]:minima[2*dim + 1]]

	# Initial parameter guesses
	mu0 = edges[max_ind]
	sigma0 = (edges[minima[2*dim + 1]] - edges[minima[2*dim]])/2
	A0 = max(Counts)*np.sqrt(np.pi)*sigma0

	try:
		# Attempt to fit a Gaussian using curve_fit
		popt, pcov = scipy.optimize.curve_fit(Gaussian, Edges, Counts,
			p0=[mu0, sigma0, A0], bounds=([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]))

		# Check goodness of fit and update the goodness variable
		if popt[0] < Edges[0] or popt[0] > Edges[-1]:
			goodness -= 1
		if popt[1] > Edges[-1] - Edges[0]:
			goodness -= 1
		if popt[2] < A0/2:
			goodness -= 1

		# Calculate parameter errors
		perr = np.sqrt(np.diag(pcov))
		for j in range(len(perr)):
			if perr[j]/popt[j] > 0.5:
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

def fit_2D(max_ind, minima, xedges, yedges, counts, gap):
	# Initialize flag and goodness variables
	flag = 1
	goodness = 11

	# Extract relevant data within the specified minima
	Xedges = xedges[minima[0]:minima[1]]
	Yedges = yedges[minima[2]:minima[3]]
	Counts = counts[minima[0]:minima[1],minima[2]:minima[3]]

	# Initial parameter guesses
	mux0 = xedges[max_ind[0]]
	muy0 = yedges[max_ind[1]]
	sigmax0 = (xedges[minima[1]] - xedges[minima[0]])/3
	sigmay0 = (yedges[minima[3]] - yedges[minima[2]])/3
	A0 = counts[max_ind[0]][max_ind[1]]

	# Create a meshgrid for fitting
	x, y = np.meshgrid(Xedges, Yedges)
	try:
		# Attempt to fit a 2D Gaussian using curve_fit
		popt, pcov = scipy.optimize.curve_fit(Gaussian_2D, (x, y), Counts.ravel(),
			p0=[mux0, muy0, sigmax0, sigmay0, A0], bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, np.inf, np.inf, np.inf]))

		# Check goodness of fit and update the goodness variable
		if popt[4] < A0/2:
			goodness -= 1
		if popt[0] < Xedges[0] or popt[0] > Xedges[-1]:
			goodness -= 1
		if popt[1] < Yedges[0] or popt[1] > Yedges[-1]:
			goodness -= 1
		if popt[2] > Xedges[-1] - Xedges[0]:
			goodness -= 1
		if popt[3] > Yedges[-1] - Yedges[0]:
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

def relabel_states(all_the_labels, states_list):
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

	for a in range(len(tmp1)):
		for b in range(len(tmp1[a])):
			tmp1[a][b] = list_unique[tmp1[a][b]]

	# Step 4: Order the states according to the mu values in the 'list1' array.
	list1.sort(key=lambda state: state.mu)

	# Create 'tmp2' by relabeling the states based on the sorted order.
	tmp2 = np.zeros_like(tmp1)
	for old_label in list_unique:
		tmp2[tmp1 == old_label] = label_to_index.get(old_label)

	return tmp2, list1

def set_final_states(list_of_states, all_the_labels, M_range):
	# Step 1: Define a criterion to determine which states are considered "final."
	# Iterate over pairs of states to compare their properties.
	old_to_new_map = []
	tmp_list = []
	for s0 in range(len(list_of_states)):
		for s1 in range(s0 + 1, len(list_of_states)):
			S0 = list_of_states[s0]
			S1 = list_of_states[s1]
			# Check whether the criteria for considering a state as "final" is met.
			if S0.peak > S1.peak and abs(S1.mu - S0.mu) < S0.sigma:
				tmp_list.append(s1)
				old_to_new_map.append([s1, s0])
			elif S0.peak < S1.peak and abs(S1.mu - S0.mu) < S1.sigma:
				tmp_list.append(s0)
				old_to_new_map.append([s0, s1])

	# Step 2: Remove states that don't meet the "final" criterion from the 'list_of_states'.
	# Note: The loop iterates in reverse to avoid index errors when removing elements.
	tmp_list = np.unique(tmp_list)[::-1]
	for s in tmp_list:
		list_of_states.pop(s)
	
	list_of_states = sorted(list_of_states, key=lambda x: x.mu)

	# Relabel accorind to the new states
	for i in range(len(all_the_labels)):
		for w in range(len(all_the_labels[i])):
			for j in range(len(old_to_new_map[::-1])):
				if all_the_labels[i][w] == old_to_new_map[j][0] + 1:
					all_the_labels[i][w] = old_to_new_map[j][1] + 1

	list_unique = np.unique(all_the_labels)
	label_to_index = {label: index for index, label in enumerate(list_unique)}
	new_labels = np.vectorize(label_to_index.get)(all_the_labels)

	# Step 3: Calculate the final threshold values and their types based on the intercept between neighboring states.
	list_of_states[0].th_inf[0] = M_range[0]
	list_of_states[0].th_inf[1] = 0

	for n in range(len(list_of_states) - 1):
		S0 = list_of_states[n]
		S1 = list_of_states[n + 1]
		a = S1.sigma**2 - S0.sigma**2
		b = -2*(S0.mu*S1.sigma**2 - S1.mu*S0.sigma**2)
		c = (S0.mu*S1.sigma)**2 - (S1.mu*S0.sigma)**2 - ((S0.sigma*S1.sigma)**2)*np.log(S0.A*S1.sigma/S1.A/S0.sigma)
		Delta = b**2 - 4*a*c
		# Determine the type of the threshold (0, 1 or 2). 
		if a == 0.0:
			th = (S0.mu + S1.mu)/2 - S0.sigma**2 / 2 / (S1.mu - S0.mu) * np.log(S0.A/S1.A)
			list_of_states[n].th_sup[0] = th
			list_of_states[n].th_sup[1] = 1
			list_of_states[n + 1].th_inf[0] = th
			list_of_states[n + 1].th_inf[1] = 1
		elif Delta >= 0:
			th_plus = (- b + np.sqrt(Delta))/(2*a)
			th_minus = (- b - np.sqrt(Delta))/(2*a)
			intercept_plus = Gaussian(th_plus, S0.mu, S0.sigma, S0.A)
			intercept_minus = Gaussian(th_minus, S0.mu, S0.sigma, S0.A)
			if intercept_plus >= intercept_minus:
				list_of_states[n].th_sup[0] = th_plus
				list_of_states[n].th_sup[1] = 1
				list_of_states[n + 1].th_inf[0] = th_plus
				list_of_states[n + 1].th_inf[1] = 1
			else:
				list_of_states[n].th_sup[0] = th_minus
				list_of_states[n].th_sup[1] = 1
				list_of_states[n + 1].th_inf[0] = th_minus
				list_of_states[n + 1].th_inf[1] = 1
		else:
			th_aver = (S0.mu/S0.sigma + S1.mu/S1.sigma)/(1/S0.sigma + 1/S1.sigma)
			list_of_states[n].th_sup[0] = th_aver
			list_of_states[n].th_sup[1] = 2
			list_of_states[n + 1].th_inf[0] = th_aver
			list_of_states[n + 1].th_inf[1] = 2

	list_of_states[-1].th_sup[0] = M_range[1]
	list_of_states[-1].th_sup[1] = 0

	# Step 4: Write the final states and final thresholds to text files.
    # The data is saved in two separate files: 'final_states.txt' and 'final_thresholds.txt'.
	with open('final_states.txt', 'w') as f:
		print('# Mu \t Sigma \t A \t state_fraction')
		for state in list_of_states:
			print(state.mu, state.sigma, state.A, state.perc, file=f)
	with open('final_thresholds.txt', 'w') as f:
		for state in list_of_states:
			print(state.th_inf[0], state.th_sup[0], file=f)

	# Step 5: Return the 'list_of_states' as the output of the function.
	return list_of_states, new_labels

def relabel_states_2D(all_the_labels, states_list):
	### Step 1: sort according to the relevance
	sorted_indices = [index + 1 for index, _ in sorted(enumerate(states_list), key=lambda x: x[1].perc, reverse=True)]
	sorted_states = sorted(states_list, key=lambda x: x.perc, reverse=True)

	### Step 2: relabel all the labels according to the new ordering
	sorted_all_the_labels = np.empty(all_the_labels.shape)
	for a, mol in enumerate(all_the_labels):
		for b, mol_t in enumerate(mol):
			for i0 in range(len(sorted_indices)):
				if mol_t == sorted_indices[i0]:
					sorted_all_the_labels[a][b] = i0 + 1
					break
				else:
					sorted_all_the_labels[a][b] = 0

	### Step 3: merge strongly overlapping states. Two states are merged if, along all the directions,
	### their means dist less than the larger of their standard deviations in that direction. 
	## Find all the pairs of states which should be merged
	merge_pairs = []
	for i, s0 in enumerate(sorted_states):
		for j, s1 in enumerate(sorted_states[i + 1:]):
			diff = np.abs(np.subtract(s1.mu, s0.mu))
			if np.all(diff < [ max(s0.sigma[k], s1.sigma[k]) for k in range(diff.size) ]):
				merge_pairs.append([i + 1, j + i + 2])

	## If a state can be merged with more than one state, choose the most relevant one
	el_to_del = []
	for p0 in range(len(merge_pairs)):
		for p1 in range(p0 + 1, len(merge_pairs)):
			if merge_pairs[p1][1] == merge_pairs[p0][1]:
				el_to_del.append(p1)
	for el in np.unique(el_to_del)[::-1]:
		merge_pairs.pop(el)

	## Manage chains of merging
	for p0 in range(len(merge_pairs)):
		for p1 in range(p0 + 1, len(merge_pairs)):
			if merge_pairs[p1][0] == merge_pairs[p0][1]:
				merge_pairs[p1][0] = merge_pairs[p0][0]

	## Create a dictionary to easily relabel data points
	state_mapping = {i: i for i in range(len(sorted_states) + 1)}
	for s0, s1 in merge_pairs:
		state_mapping[s1] = s0

	## Relabel the data points
	updated_labels = np.empty(sorted_all_the_labels.shape)
	for a, mol in enumerate(sorted_all_the_labels):
		for b, label in enumerate(mol):
			try:
				updated_labels[a][b] = state_mapping[label]
			except:
				continue
				# print('No classification found.')

	## Update the list of states
	states_to_remove = set(s1 for s0, s1 in merge_pairs)
	updated_states = [sorted_states[s] for s in range(len(sorted_states)) if s + 1 not in states_to_remove]

	### Step 4: remove gaps in the labeling
	current_labels = np.unique(updated_labels)
	if current_labels[0] != 0:
		np.insert(current_labels, 0, 0)
	for i, mol in enumerate(updated_labels):
		for t, l in enumerate(mol):
			for m in range(len(current_labels)):
				if l == current_labels[m]:
					updated_labels[i][t] = m

	for s_id in range(len(updated_states)):
		n = np.sum(updated_labels == s_id+1)
		updated_states[s_id].perc = n / updated_labels.size

	### Step 5: print informations on the final states
	with open('final_states.txt', 'w') as f:
		print('#center_coords, semiaxis, fraction_of_data', file=f)
		for s in updated_states:
			C = s.mu
			centers = '[' + str(C[0]) + ', '
			for ck in C[1:-1]:
				centers += str(ck) + ', '
			centers += str(C[-1]) + ']'
			A = s.a
			axis = '[' + str(A[0]) + ', '
			for ck in A[1:-1]:
				axis += str(ck) + ', '
			axis += str(A[-1]) + ']'
			print(centers, axis, s.perc, file=f)

	return updated_labels, updated_states

def assign_single_frames(all_the_labels, tau_window):
	print('* Assigning labels to the single frames...')
	new_labels = np.repeat(all_the_labels, tau_window, axis=1)
	return new_labels

def plot_TRA_figure(number_of_states, fraction_0, PAR):
	t_conv, units = PAR.t_conv, PAR.t_units
	number_of_states = np.array(number_of_states)
	x = np.array(number_of_states.T[0])*t_conv
	number_of_states = number_of_states[:, 1:]
	fraction_0 = np.array(fraction_0)[:, 1:]

	fig, ax = plt.subplots()
	### If I want to chose one particular value of the smoothing: #########
	t_smooth_idx = 1
	y = number_of_states.T[t_smooth_idx]
	y2 = fraction_0.T[t_smooth_idx]
	#######################################################################

	# ### If I want to average over the different smoothings: ###############
	# y = np.mean(number_of_states, axis=1)
	# y_err = np.std(number_of_states, axis=1)
	# y_inf = y - y_err
	# y_sup = y + y_err
	# ax.fill_between(x, y_inf, y_sup, zorder=0, alpha=0.4, color='gray')
	# y2 = np.mean(fraction_0, axis=1)
	# #######################################################################

	### General plot settings ###
	ax.plot(x, y, marker='o')
	ax.set_xlabel(r'Time resolution $\Delta t$ ' + units)#, weight='bold')
	ax.set_ylabel(r'# environments', weight='bold', c='#1f77b4')
	ax.set_xscale('log')
	ax.set_xlim(x[0]*0.75, x[-1]*1.5)
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))

	### Top x-axes settings ###
	ax2 = ax.twiny()
	ax2.set_xlabel(r'Time resolution $\Delta t$ [frames]')#, weight='bold')
	ax2.set_xscale('log')
	ax2.set_xlim(x[0]*0.75/t_conv, x[-1]*1.5/t_conv)

	axr = ax.twinx()
	axr.plot(x, y2, marker='o', c='#ff7f0e')
	axr.set_ylabel('Population of env 0', weight='bold', c='#ff7f0e')

	plt.show()
	fig.savefig('Time_resolution_analysis.png', dpi=600)

def print_mol_labels_fbf_gro(all_the_labels):
	print('* Print color IDs for Ovito...')
	with open('all_cluster_IDs_gro.dat', 'w') as f:
		for labels in all_the_labels:
			# Join the elements of 'labels' using a space as the separator and write to the file.
			print(' '.join(map(str, labels)), file=f)

def print_mol_labels_fbf_xyz(all_the_labels):
	print('* Print color IDs for Ovito...')
	with open('all_cluster_IDs_xyz.dat', 'w') as f:
		for t in range(all_the_labels.shape[1]):
			# Print two lines containing '#' to separate time steps.
			print('#', file=f)
			print('#', file=f)
			# Use np.savetxt to write the labels for each time step efficiently.
			np.savetxt(f, all_the_labels[:, t], fmt='%d', comments='')

def print_mol_labels_fbf_lam(all_the_labels):
	print('* Print color IDs for Ovito...')
	with open('all_cluster_IDs_lam.dat', 'w') as f:
		for t in range(all_the_labels.shape[1]):
			# Print nine lines containing '#' to separate time steps.
			for k in range(9):
				print('#', file=f)
			# Use np.savetxt to write the labels for each time step efficiently.
			np.savetxt(f, all_the_labels[:, t], fmt='%d', comments='')

def print_colored_trj_from_xyz(trj_file, all_the_labels, PAR):
	if os.path.exists(trj_file):
		with open(trj_file, "r") as f:
			tmp = [ x.split()  for x in f.readlines() ]

		N = all_the_labels.shape[0]
		T = all_the_labels.shape[1]
		nlines = (N + 2)*T

		print('\t Removing the first', int(PAR.t_smooth/2) + PAR.t_delay, 'frames...')
		for t in range(int(PAR.t_smooth/2) + PAR.t_delay):
			for n in range(N + 2):
				tmp.pop(0)

		print('\t Removing the last', int((len(tmp) - nlines)/(N + 2)), 'frames...')
		while len(tmp) > nlines:
			tmp.pop(-1)

		with open('colored_trj.xyz', "w+") as f:
			i = 0
			for t in range(T):
				print(tmp[i][0], file=f)
				print(tmp[i + 1][0], file=f)
				for n in range(N):
					print(all_the_labels[n][t], tmp[i + 2 + n][1], tmp[i + 2 + n][2], tmp[i + 2 + n][3], file=f)
				i += N + 2
	else:
		print('No ' + trj_file + ' found for coloring the trajectory.')

