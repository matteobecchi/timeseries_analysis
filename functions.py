import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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
			param = [float(line.strip()) for line in lines]
	except:
		print('\tinput_parameters.txt file missing or wrongly formatted.')

	# Step 4: Create a list containing the extracted parameters, converting them to integers where needed.
	# The fifth parameter, 'bins' is optional.  
	if len(param) == 4:
		PAR = [int(param[0]), int(param[1]), param[2], int(param[3])]
	elif len(param) == 5:
		print('\tWARNING: overriding histogram binning')
		PAR = [int(param[0]), int(param[1]), param[2], int(param[3]), int(param[4])]

	print('Reading data from', data_dir)

	# Step 5: Check if the shape of 'data_dir' array is (2, ).
	# If yes, return 'data_dir' as an array along with 'PAR'.
	# Otherwise, return 'data_dir' as a string along with 'PAR'.
	if data_dir.shape == (2, ):
		return data_dir, PAR
	else:
		return str(data_dir), PAR

def read_data(filename):
	print('* Reading data...')
	# Step 1: Check if the filename ends with '.npz', indicating a NumPy binary file.
	if filename.endswith('.npz'):
		# Step 2: Load the data from the '.npz' file using a context manager (to automatically close the file afterward).
		with np.load(filename) as data:
			lst = data.files  # Get the list of variable names saved in the '.npz' file.
			M = np.array(data[lst[0]])  # Load the first variable (assumed to be the data) into a NumPy array.
			# Step 3: Check if the data array has three dimensions.
			if M.ndim == 3:
				# If the data has three dimensions, stack it along the first axis to make it two-dimensional.
				M = np.vstack(M)
				# Transpose the data to have the desired shape (assumed to be (2048, num_samples)).
				M = M.T
			# Step 4: Check if the number of rows in the data array is not equal to 2048.
			if M.shape[0] != 2048:
				# If the number of rows is not 2048, transpose the data array to make it compatible.
				M = M.T
			return M
	# Step 5: Check if the filename ends with '.npy', indicating a NumPy binary file.
	elif filename.endswith('.npy'):
		# Step 6: Load the data from the '.npy' file directly into a NumPy array.
		M = np.load(filename)
		return M
	# If the file format is not supported, print an error message and return None.
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
	weights = np.repeat(1.0, window) / window

	# Step 2: Apply the moving average filter to the 'data' array using the 'weights' array.
	# The 'np.convolve' function performs a linear convolution between 'data' and 'weights'.
	# The result is a smoothed version of the 'data', where each point represents the weighted average of its neighbors.
	# Mode ‘valid’ returns output of length max(M, N) - min(M, N) + 1. 
	# The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect.
	# If data.ndim == 2, return the smoothing on the columns
	if data.ndim == 1:
		return np.convolve(data, weights, mode='valid')
	elif data.ndim == 2:
		tmp = []
		for x in data:
			tmp.append(np.convolve(x, weights, mode='valid'))
		return np.array(tmp)
	else:
		print('\tERROR: impossible to performe moving average on the argument.')
		return []

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

def relabel_states(all_the_labels, list_of_states):
	# Step 1: Remove empty states from the 'list_of_states' and keep only non-empty states.
	# A non-empty state is one where the third element (index 2) is not equal to 0.0.
	list1 = [state for state in list_of_states if state[2] != 0.0]

	# Step 2: Get the unique labels from the 'all_the_labels' array. 
	list_unique = np.unique(all_the_labels)

	# Step 3: Relabel the states from 0 to n_states-1 based on their occurrence in 'all_the_labels'.
	tmp1 = all_the_labels.copy()  # Create a copy of the 'all_the_labels' array to work with.
	for i, l in enumerate(list_unique):
		# Loop over each unique label 'l'.
		for a in range(len(all_the_labels)):
			for b in range(len(all_the_labels[a])):
				if all_the_labels[a][b] == l:
					tmp1[a][b] = i

	# Step 4: Order the states according to the mu values in the 'list1' array.
	# Get the mu values from each state in 'list1' and store them in 'list_of_mu'.
	list_of_mu = np.array([state[0][0] for state in list1])
	# Create a copy of 'list_of_mu' to use for sorting while keeping the original 'list_of_mu' unchanged.
	copy_of_list_of_mu = list_of_mu.copy()
	# 'sorted_IDs' will store the sorted indices of 'list1' based on the mu values.
	sorted_IDs = []

	while len(copy_of_list_of_mu) > 0:
		# Find the minimum mu value and its index in the 'copy_of_list_of_mu'.
		min_mu = np.min(copy_of_list_of_mu)
		ID_min = np.where(list_of_mu == min_mu)[0][0]
		sorted_IDs.append(ID_min)
		# Remove the minimum mu value from 'copy_of_list_of_mu'.
		ID_min2 = np.where(copy_of_list_of_mu == min_mu)[0][0]
		copy_of_list_of_mu = np.delete(copy_of_list_of_mu, ID_min2)

	# Reorder the states in 'list1' and 'tmp1' based on 'sorted_IDs'.
	list2 = [list1[ID] for ID in sorted_IDs]
	tmp2 = np.zeros_like(tmp1)

	for i, l in enumerate(sorted_IDs):
		for a in range(len(tmp1)):
			for b in range(len(tmp1[a])):
				if tmp1[a][b] == l:
					tmp2[a][b] = i

	return tmp2, list2

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

	# Step 3: Create a new list 'final_list' to store the final threshold values and their types (0, 1, or 2).
	final_list = []
	final_list.append([0.0, 0])  # Initialize the list with the starting threshold.

	mu = np.array([state[0][0] for state in list_of_states])
	sigma = np.array([state[0][1] for state in list_of_states])
	A = np.array([state[0][2] for state in list_of_states])
	peak = A/sigma/np.sqrt(np.pi)

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
	out_of_bounds = []
	for n in range(len(final_list)):
		if final_list[n][0] < 0.0 or final_list[n][0] > 1.0:
			out_of_bounds.append(n)
	for index in out_of_bounds[::-1]:
		del final_list[index]

	# Step 5: Sort the thresholds and add missing states.
	final_list = np.array(final_list)
	final_list = sorted(final_list, key=lambda x: x[0])
	tmp_list_of_states = []
	m = 0
	for n in range(len(final_list) - 1):
		if list_of_states[m][0][0] > final_list[n][0] and list_of_states[m][0][0] < final_list[n + 1][0]:
			tmp_list_of_states.append(list_of_states[m])
			m += 1
		else:
			if list_of_states[m][0][0] >= final_list[n + 1][0]:
				new_mu = (final_list[n][0] + final_list[n - 1][0])/2
				new_sigma = (final_list[n][0] - final_list[n - 1][0])/2
				new_A = 1.0
				tmp_list_of_states.append([[new_mu, new_sigma, new_A], [final_list[n][0], final_list[n + 1][0]], 1.0])
			else:
				new_mu = (final_list[n][0] + final_list[n + 1][0])/2
				new_sigma = (final_list[n + 1][0] - final_list[n][0])/2
				new_A = 1.0
				tmp_list_of_states.append([[new_mu, new_sigma, new_A], [final_list[n][0], final_list[n + 1][0]], 1.0])
			m += 1
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

def print_mol_labels_fbf_gro(all_the_labels, filename):
	print('* Print color IDs for Ovito...')
	with open(filename, 'w') as f:
		for labels in all_the_labels:
			# Join the elements of 'labels' using a space as the separator and write to the file.
			print(' '.join(map(str, labels)), file=f)

def print_mol_labels_fbf_xyz(all_the_labels, filename):
	print('* Print color IDs for Ovito...')
	with open(filename, 'w') as f:
		for t in range(all_the_labels.shape[1]):
			# Print two lines containing '#' to separate time steps.
			print('#', file=f)
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

