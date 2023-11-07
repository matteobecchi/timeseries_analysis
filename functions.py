import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import copy
import math
import scipy.optimize
from scipy.signal import savgol_filter
from matplotlib.colors import LogNorm
import plotly
plotly.__version__
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import seaborn as sns

def read_input_parameters():
	# Step 1: Attempt to read the content of 'data_directory.txt' file and load it into a NumPy array as strings.
	try:
		data_dir = np.loadtxt('data_directory.txt', dtype=str)
	except:
		print('\tdata_directory.txt file missing or wrongly formatted.')

	# Step 2: Attempt to open and read the 'input_parameters.txt' file.
	try:
		with open('input_parameters.txt', 'r') as file:
			lines = file.readlines()
			# Step 3: Convert the lines of text into a list of floating-point numbers (floats).
			param = [line.strip() for line in lines]
	except:
		print('\tinput_parameters.txt file missing or wrongly formatted.')

	# Step 4: Create a list containing the extracted parameters, converting them to integers where needed.
	# The sixth parameter, 'bins' is optional and shoul be avoided if possible. 
	if len(param) == 6:
		PAR = [int(param[0]), int(param[1]), int(param[2]), float(param[3]), r'[' + str(param[4]) + r']',  int(param[5])]
	elif len(param) == 7:
		print('\tWARNING: overriding histogram binning')
		PAR = [int(param[0]), int(param[1]), int(param[2]), float(param[3]), r'[' + str(param[4]) + r']', int(param[5]), int(param[6])]
	else:
		print('\tinput_parameters.txt file wrongly formatted.')

	print('* Reading data from', data_dir)

	if data_dir.size == 1:
		return str(data_dir), PAR
	else:
		return data_dir, PAR		

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

def relabel_states(all_the_labels, list_of_states):
	# Step 1: Remove empty states from the 'list_of_states' and keep only non-empty states.
	# A non-empty state is one where the third element (index 2) is not equal to 0.0.
	list1 = [state for state in list_of_states if state[2] != 0.0]

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
	list1.sort(key=lambda state: state[0][0])

	# Create 'tmp2' by relabeling the states based on the sorted order.
	tmp2 = np.zeros_like(tmp1)
	for old_label in list_unique:
		tmp2[tmp1 == old_label] = label_to_index.get(old_label)

	return tmp2, list1

def set_final_states(list_of_states, all_the_labels, M_range):
	old_to_new_map = []
	# Step 1: Define a criterion to determine which states are considered "final."
	# Iterate over pairs of states to compare their properties (mu, sigma, and amplitude).

	mu = np.array([state[0][0] for state in list_of_states])
	sigma = np.array([state[0][1] for state in list_of_states])
	A = np.array([state[0][2] for state in list_of_states])
	peak = A/sigma/np.sqrt(np.pi)

	tmp_list = []
	for s0 in range(len(list_of_states)):
		for s1 in range(s0 + 1, len(list_of_states)):
			# Check whether the criteria for considering a state as "final" is met.
			if peak[s0] > peak[s1] and abs(mu[s1] - mu[s0]) < sigma[s0]:
				tmp_list.append(s1)
				old_to_new_map.append([s1, s0])
			elif peak[s0] < peak[s1] and abs(mu[s1] - mu[s0]) < sigma[s1]:
				tmp_list.append(s0)
				old_to_new_map.append([s0, s1])

	# Step 2: Remove states that don't meet the "final" criterion from the 'list_of_states'.
	# Note: The loop iterates in reverse to avoid index errors when removing elements.
	tmp_list = np.unique(tmp_list)[::-1]
	for s in tmp_list:
		list_of_states.pop(s)
	
	list_of_states = sorted(list_of_states, key=lambda x: x[0][0])

	# Relabel accorind to the new states
	for i in range(len(all_the_labels)):
		for w in range(len(all_the_labels[i])):
			for j in range(len(old_to_new_map[::-1])):
				if all_the_labels[i][w] == old_to_new_map[j][0] + 1:
					all_the_labels[i][w] = old_to_new_map[j][1] + 1

	list_unique = np.unique(all_the_labels)
	label_to_index = {label: index for index, label in enumerate(list_unique)}
	new_labels = np.vectorize(label_to_index.get)(all_the_labels)

	# Step 3: Create a new list 'final_list' to store the final threshold values and their types (0, 1, 2 or 3).
	final_list = []
	final_list.append([M_range[0], 0])  # Initialize the list with the starting threshold.

	mu = np.array([state[0][0] for state in list_of_states])
	sigma = np.array([state[0][1] for state in list_of_states])
	A = np.array([state[0][2] for state in list_of_states])
	peak = A / sigma / np.sqrt(np.pi)

	# Step 4: Calculate the final threshold values and their types based on the intercept between neighboring states.
	for s in range(len(list_of_states) - 1):
		a = sigma[s + 1]**2 - sigma[s]**2
		b = -2*(mu[s]*sigma[s + 1]**2 - mu[s + 1]*sigma[s]**2)
		c = (mu[s]*sigma[s + 1])**2 - (mu[s + 1]*sigma[s])**2 - ((sigma[s]*sigma[s + 1])**2)*np.log(A[s]*sigma[s + 1]/A[s + 1]/sigma[s])
		Delta = b**2 - 4*a*c
		# Determine the type of the threshold (0, 1, 2 or 3). 
		if Delta >= 0:
			th_plus = (- b + np.sqrt(Delta))/(2*a)
			th_minus = (- b - np.sqrt(Delta))/(2*a)
			intercept_plus = Gaussian(th_plus, mu[s], sigma[s], A[s])
			intercept_minus = Gaussian(th_minus, mu[s], sigma[s], A[s])
			if intercept_plus >= intercept_minus:
				final_list.append([th_plus, 1])
			else:
				final_list.append([th_minus, 1])
		else:
			final_list.append([(mu[s]/sigma[s] + mu[s + 1]/sigma[s + 1])/(1/sigma[s] + 1/sigma[s + 1]), 3])
	final_list.append([M_range[1], 0])

	# Remove the tresholds outside the interval [0, 1]
	final_list = [entry for entry in final_list if M_range[0] <= entry[0] <= M_range[1]]

	# Step 5: Sort the thresholds.
	final_list = np.array(final_list)
	final_list = sorted(final_list, key=lambda x: x[0])

	# Step 6: Write the final states and final thresholds to text files.
    # The data is saved in two separate files: 'final_states.txt' and 'final_thresholds.txt'.
	with open('final_states.txt', 'w') as f:
		for state in list_of_states:
			print(state[0][0], state[0][1], state[0][2], file=f)
	with open('final_thresholds.txt', 'w') as f:
		for th in final_list:
			print(th[0], file=f)

	# Step 7: Return the 'list_of_states' and 'final_list' as the output of the function.
	return list_of_states, final_list, new_labels

def relabel_states_2D(all_the_labels, list_of_states):
	### Step 1: sort according to the relevance, and remove possible empty states
	sorted_indices = [index + 1 for index, _ in sorted(enumerate(list_of_states), key=lambda x: x[1][2], reverse=True)]
	sorted_states = sorted(list_of_states, key=lambda x: x[2], reverse=True)

	# Step 2: relabel all the labels according to the new ordering
	sorted_all_the_labels = np.empty(all_the_labels.shape)
	for a, mol in enumerate(all_the_labels):
		for b, mol_t in enumerate(mol):
			for i0 in range(len(sorted_indices)):
				if mol_t == sorted_indices[i0]:
					sorted_all_the_labels[a][b] = i0 + 1
					break
				else:
					sorted_all_the_labels[a][b] = 0

	# Step 3: merge strongly overlapping states
	merge_pairs = []
	for i, s0 in enumerate(sorted_states):
		for j, s1 in enumerate(sorted_states[i + 1:]):
			C0 = s0[1][0]
			C1 = s1[1][0]
			diff = np.abs(np.subtract(C1, C0))
			if np.all(diff < [ max(s0[1][1][k], s1[1][1][k]) for k in range(diff.size) ]):
				merge_pairs.append([i + 1, j + i + 2])

	for p0 in range(len(merge_pairs)):
		for p1 in range(p0 + 1, len(merge_pairs)):
			if merge_pairs[p1][0] == merge_pairs[p0][1]:
				merge_pairs[p1][0] = merge_pairs[p0][0]

	state_mapping = {i: i for i in range(len(sorted_states) + 1)}
	for s0, s1 in merge_pairs:
		state_mapping[s1] = s0

	updated_labels = np.empty(sorted_all_the_labels.shape)
	for a, mol in enumerate(sorted_all_the_labels):
		for b, label in enumerate(mol):
			try:
				updated_labels[a][b] = state_mapping[label]
			except:
				print('No classification found.')
	states_to_remove = set(s1 for s0, s1 in merge_pairs)
	updated_states = [sorted_states[s] for s in range(len(sorted_states)) if s + 1 not in states_to_remove]

	# Step 4: remove gaps in the labeling
	current_labels = np.unique(updated_labels)
	for i, mol in enumerate(updated_labels):
		for t, l in enumerate(mol):
			for m in range(len(current_labels)):
				if l == current_labels[m]:
					updated_labels[i][t] = m

	# Step 5: print informations on the final states
	with open('final_states.txt', 'w') as f:
		print('#center_coords, semiaxis', file=f)
		for s in updated_states:
			C = s[1][0]
			centers = '[' + str(C[0]) + ', '
			for ck in C[1:-1]:
				centers += str(ck) + ', '
			centers += str(C[-1]) + ']'
			A = s[1][1]
			axis = '[' + str(A[0]) + ', '
			for ck in A[1:-1]:
				axis += str(ck) + ', '
			axis += str(A[-1]) + ']'
			print(centers, axis, file=f)

	return updated_labels, updated_states

def assign_single_frames(all_the_labels, tau_window):
	print('* Assigning labels to the single frames...')
	new_labels = np.repeat(all_the_labels, tau_window, axis=1)
	return new_labels

def plot_TRA_figure(number_of_states, fraction_0, tau_window, t_conv, units, filename):
	fig, ax = plt.subplots()

	x = np.array(tau_window)*t_conv

	### If I want to chose one particular value of the smoothing: #########
	y = number_of_states.T[0]
	y2 = fraction_0.T[0]
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
	fig.savefig(filename + '.png', dpi=600)

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

