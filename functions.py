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
	# The fifth parameter, 'bins' is optional and shoul be avoided if possible. 
	if len(param) == 5:
		PAR = [int(param[0]), int(param[1]), float(param[2]), r'[' + str(param[3]) + r']',  int(param[4])]
	elif len(param) == 6:
		print('\tWARNING: overriding histogram binning')
		PAR = [int(param[0]), int(param[1]), float(param[2]), r'[' + str(param[3]) + r']', int(param[4]), int(param[5])]
	else:
		print('\tinput_parameters.txt file wrongly formatted.')

	print('* Reading data from', data_dir)

	# Step 5: Check if the shape of 'data_dir' array is (2, ).
	# If yes, return 'data_dir' as an array along with 'PAR'.
	# Otherwise, return 'data_dir' as a string along with 'PAR'.
	if data_dir.shape == (2, ):
		return data_dir, PAR
	else:
		return str(data_dir), PAR

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
	elif data.ndim == 2:
		return np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'), axis=1, arr=data)
	else:
		raise ValueError('Invalid array dimension. Only 1D and 2D arrays are supported.')

def moving_average_2D(data, L):
	if L%2 == 0:
		print('\tL must be an odd number.')
		return
	l = int((L - 1)/2)
	tmp = np.empty(data.shape)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			i0 = np.max([i - l, 0])
			i1 = np.min([i + l + 1, data.shape[0]])
			j0 = np.max([j - l, 0])
			j1 = np.min([j + l + 1, data.shape[1]])
			square = data[i0:i1,j0:j1]
			if square.size > 0:
				tmp[i][j] = np.sum(square)/square.size
			else:
				tmp[i][j] = 0.0
	return np.array(tmp)

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

def fit_2D(max_ind, minima, xedges, yedges, counts, gap):
		flag = 1
		goodness = 11
		Xedges = xedges[minima[0]:minima[1]]
		Yedges = yedges[minima[2]:minima[3]]
		Counts = counts[minima[0]:minima[1],minima[2]:minima[3]]
		mux0 = xedges[max_ind[0]]
		muy0 = yedges[max_ind[1]]
		sigmax0 = (xedges[minima[1]] - xedges[minima[0]])/3
		sigmay0 = (yedges[minima[3]] - yedges[minima[2]])/3
		A0 = counts[max_ind[0]][max_ind[1]]
		x, y = np.meshgrid(Xedges, Yedges)
		try:
			popt, pcov = scipy.optimize.curve_fit(Gaussian_2D, (x, y), Counts.ravel(),
				p0=[mux0, muy0, sigmax0, sigmay0, A0], bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, np.inf, np.inf, np.inf]))
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
			perr = np.sqrt(np.diag(pcov))
			for j in range(len(perr)):
				if perr[j]/popt[j] > 0.5:
					goodness -= 1
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

	# Step 4: Order the states according to the mu values in the 'list1' array.
	list1.sort(key=lambda state: state[0][0])

	# # Create 'tmp2' by relabeling the states based on the sorted order.
	tmp2 = np.zeros_like(tmp1)
	for old_label in list_unique:
		tmp2[tmp1 == old_label] = label_to_index.get(old_label)

	return tmp2, list1

def set_final_states(list_of_states):
	# Step 1: Define an arbitrary criterion to determine which states are considered "final."
	# Iterate over pairs of states to compare their properties (mu, sigma, and amplitude).

	mu = np.array([state[0][0] for state in list_of_states])
	sigma = np.array([state[0][1] for state in list_of_states])
	A = np.array([state[0][2] for state in list_of_states])
	peak = A/sigma/np.sqrt(np.pi)

	tmp_list = []
	for s0 in range(len(list_of_states)):
		for s1 in range(s0 + 1, len(list_of_states)):
			# Check whether the criteria for considering a state as "final" is met.
			if peak[s0] > peak[s1] and mu[s1] - mu[s0] < sigma[s0]:
				tmp_list.append(s1)
			elif peak[s0] < peak[s1] and mu[s1] - mu[s0] < sigma[s1]:
				tmp_list.append(s0)

	# Step 2: Remove states that don't meet the "final" criterion from the 'list_of_states'.
	# Note: The loop iterates in reverse to avoid index errors when removing elements.
	for s in np.unique(tmp_list)[::-1]:
		list_of_states.pop(s)

	# Step 3: Create a new list 'final_list' to store the final threshold values and their types (0, 1, 2 or 3).
	final_list = []
	final_list.append([0.0, 0])  # Initialize the list with the starting threshold.

	mu = np.array([state[0][0] for state in list_of_states])
	sigma = np.array([state[0][1] for state in list_of_states])
	A = np.array([state[0][2] for state in list_of_states])
	peak = A / sigma / np.sqrt(np.pi)

	# Step 4: Calculate the final threshold values and their types based on the intercept between neighboring states.
	for s in range(len(list_of_states) - 1):
		a = sigma[s + 1]**2 - sigma[s]**2
		b = -2*(mu[s]*sigma[s + 1]**2 - mu[s + 1]*sigma[s]**2)
		c = (mu[s]*sigma[s + 1])**2 - (mu[s + 1]*sigma[s])**2 - (sigma[s]*sigma[s + 1])**2*np.log(A[s]*sigma[s + 1]/A[s + 1]/sigma[s])
		Delta = b**2 - 4*a*c
		# Determine the type of the threshold (0, 1, 2 or 3). 
		if Delta >= 0:
			th_plus = (- b + np.sqrt(Delta))/(2*a)
			th_minus = (- b - np.sqrt(Delta))/(2*a)
			intercept_plus = Gaussian(th_plus, mu[s], sigma[s], A[s])
			intercept_minus = Gaussian(th_minus, mu[s], sigma[s], A[s])
			if intercept_plus >= intercept_minus:
				final_list.append([th_plus, 1])
				if Gaussian(th_minus, mu[s], sigma[s], A[s]) > 0.01*np.max(peak):
					final_list.append([th_minus, 2])
			else:
				final_list.append([th_minus, 1])
				if Gaussian(th_plus, mu[s], sigma[s], A[s]) > 0.01*np.max(peak):
					final_list.append([th_plus, 2])
		else:
			final_list.append([(mu[s]/sigma[s] + mu[s + 1]/sigma[s + 1])/(1/sigma[s] + 1/sigma[s + 1]), 3])
	final_list.append([1.0, 0])

	# Remove the tresholds outside the interval [0, 1]
	final_list = [entry for entry in final_list if 0.0 <= entry[0] <= 1.0]

	# Step 5: Sort the thresholds and add missing states.
	final_list = np.array(final_list)
	final_list = sorted(final_list, key=lambda x: x[0])
	tmp_list_of_states = []

	for n in range(len(final_list) - 1):
		possible_states = []
		for state in list_of_states:
			if state[0][0] > final_list[n][0] and state[0][0] < final_list[n + 1][0]:
				possible_states.append(state)
		if len(possible_states) > 0:
			biggest_state = max(possible_states, key = lambda element: element[0][2])
			tmp_list_of_states.append(biggest_state)
		else:
			new_mu = (final_list[n][0] + final_list[n + 1][0])/2
			new_sigma = (final_list[n + 1][0] - final_list[n][0])/2
			new_A = 1.0
			tmp_list_of_states.append([[new_mu, new_sigma, new_A], [final_list[n][0], final_list[n + 1][0]], 1.0])
	list_of_states = tmp_list_of_states

	# Step 6: Write the final states and final thresholds to text files.
    # The data is saved in two separate files: 'final_states.txt' and 'final_thresholds.txt'.
	with open('final_states.txt', 'w') as f:
		for state in list_of_states:
			print(state[0][0], state[0][1], state[0][2], file=f)
	with open('final_tresholds.txt', 'w') as f:
		for th in final_list:
			print(th[0], file=f)

	# Step 7: Return the 'list_of_states' and 'final_list' as the output of the function.
	return list_of_states, final_list

def relabel_states_2D(all_the_labels, list_of_states):
	# print(np.unique(all_the_labels), len(list_of_states))
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
			if (diff[0] < np.max([s0[0][2], s1[0][2]]) and diff[1] < np.max([s0[0][3], s1[0][3]])):
				merge_pairs.append([i + 1, j + i + 2])

	state_mapping = {i: i for i in range(len(sorted_states) + 1)}
	for s0, s1 in merge_pairs:
		state_mapping[s1] = s0
	updated_labels = np.empty(sorted_all_the_labels.shape)
	for a, mol in enumerate(sorted_all_the_labels):
		for b, label in enumerate(mol):
			updated_labels[a][b] = state_mapping[label]
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
		print('#[Cx, Cy] a b', file=f)
		for s in updated_states:
			print('[' + str(s[0][0]) + ', ' +  str(s[0][1]) + ']', s[0][2], s[0][3], file=f)

	return updated_labels, updated_states

def assign_final_states_to_single_frames(M, final_list):
	print('* Assigning labels to the single frames...')
	# Create an array of threshold values for comparison (shape: (1, len(final_list))).
	thresholds = np.array([item[0] for item in final_list])
	# Create a mask to compare M with the thresholds (shape: (M.shape[0], M.shape[1], len(final_list))).
	mask = (M[:, :, np.newaxis] >= thresholds[:-1]) & (M[:, :, np.newaxis] <= thresholds[1:])
	# Assign labels using argmax to find the index of the first True value along the third axis.
	all_the_labels = np.argmax(mask, axis=2)
	# In case a value in M is outside the threshold range, the last label will be selected (len(final_list) - 1).
	all_the_labels[~mask.any(axis=2)] = len(final_list) - 1
	return all_the_labels

def assign_final_states_to_single_frames_2D(M, final_list):
	print('* Assigning labels to the single frames...')
	# Create an array with the centers of the final states
	ell_centers = np.array([ state[1][0] for state in final_list ])
	ell_axes = np.array([ [state[1][1], state[1][2]] for state in final_list ])
	# For every point, compute the distance from all the centers
	M_expanded = M[:, :, np.newaxis, :]
	centers_expanded = ell_centers[np.newaxis, np.newaxis, :, :]
	axes_expanded = ell_axes[np.newaxis, np.newaxis, :, :]
	D = np.sum(((M_expanded - centers_expanded)/axes_expanded) ** 2, axis=3)
	# Find the center closest to each point. That will be its cluster. 
	all_the_labels = np.argmin(D, axis=2)
	return all_the_labels

def print_mol_labels1(all_the_labels, PAR, filename):
	print('* Print color IDs for Ovito...')
	tau_window = PAR[0]
	with open(filename, 'w') as f:
		for i in range(all_the_labels.shape[0]):
			string = str(all_the_labels[i][0])
			for t in range(1, tau_window):
					string += ' ' + str(all_the_labels[i][0])
			string = ' '.join([(str(label) + ' ') * tau_window for label in all_the_labels[i][1:]])
			print(string, file=f)

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
	with open('all_cluster_IDs_xyz.dat', 'w') as f:
		for t in range(all_the_labels.shape[1]):
			# Print nine lines containing '#' to separate time steps.
			for k in range(9):
				print('#', file=f)
			# Use np.savetxt to write the labels for each time step efficiently.
			np.savetxt(f, all_the_labels[:, t], fmt='%d', comments='')

def tmp_print_some_data(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	with open(filename, 'w') as f:
		with open('labels_for_PCA.txt', 'w') as f2:
			print('### Size of the time window: ' + str(tau_window) + ' frames. ', file=f)
			for i in range(all_the_labels.shape[0]):
				for w in range(all_the_labels.shape[1]):
					if all_the_labels[i][w] > 1:
						print(all_the_labels[i][w], file=f2)
						for t in range(tau_window):
							print(M[i][w*tau_window + t], file=f)
	with open('for_Martina_PCA_ALL.txt', 'w') as f:
		print('### Size of the time window: ' + str(tau_window) + ' frames. ', file=f)
		for i in range(all_the_labels.shape[0]):
			for w in range(all_the_labels.shape[1]):
				for t in range(tau_window):
					print(M[i][w*tau_window + t], file=f)

def letter_subplots(axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs):
    """Add letters to the corners of subplots (panels). By default each axis is
    given an uppercase bold letter label placed in the upper-left corner.
    Args
        axes : list of pyplot ax objects. default plt.gcf().axes.
        letters : list of strings to use as labels, default ["A", "B", "C", ...]
        xoffset, yoffset : positions of each label relative to plot frame
          (default -0.1,1.0 = upper left margin). Can also be a list of
          offsets, in which case it should be the same length as the number of
          axes.
        Other keyword arguments will be passed to annotate() when panel letters
        are added.
    Returns:
        list of strings for each label added to the axes
    Examples:
        Defaults:
            >>> fig, axes = plt.subplots(1,3)
            >>> letter_subplots() # boldfaced A, B, C
        
        Common labeling schemes inferred from the first letter:
            >>> fig, axes = plt.subplots(1,4)        
            >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
        Fully custom lettering:
            >>> fig, axes = plt.subplots(2,1)
            >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
        Per-axis offsets:
            >>> fig, axes = plt.subplots(1,2)
            >>> letter_subplots(axes, xoffset=[-0.1, -0.15])
            
        Matrix of axes:
            >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
            >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix
    """

    # get axes:
    if axes is None:
        axes = plt.gcf().axes
    # handle single axes:
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)])
    llets = list('abcdefghijklmnopqrstuvwxyz'[:len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [ "({})".format(lett) for lett in llets ]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [ "({})".format(lett) for lett in ulets ]
        fontweight = "normal"
    elif letters in ("lower", "lowercase", "a"):
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset]*len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset]*len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(fontweight=fontweight, fontsize='large', ha="center",
                       va='center', xycoords='axes fraction', annotation_clip=False)
    kwargs = dict( list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax,lbl,xoff,yoff in zip(axes,letters,xoffset,yoffset):
        t = ax.annotate(lbl, xy=(xoff,yoff), **kwargs)
        list_txts.append(t)
    return list_txts
