import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_data(filename):
	# Step 1: Check if the filename ends with '.npz', indicating a NumPy binary file.
	if filename.endswith('.npz'):
		# Step 2: Load the data from the '.npz' file using a context manager (to automatically close the file afterward).
		with np.load(filename) as data:
			lst = data.files  # Get the list of variable names saved in the '.npz' file.
			M = np.array(data[lst[0]])  # Load the first variable (assumed to be the data) into a NumPy array.
			# Step 3: Check if the data array has three dimensions.
			print(M.shape)
			if M.ndim == 3:
				# If the data has three dimensions, stack it along the first axis to make it two-dimensional.
				M = np.vstack(M)
				# Transpose the data to have the desired shape (assumed to be (2048, num_samples)).
				M = M.T
			# Step 4: Check if the number of rows in the data array is not equal to 2048.
			if M.shape[0] != 2048:
				# If the number of rows is not 2048, transpose the data array to make it compatible.
				M = M.T
			print('\tOriginal data shape:', M.shape)
			return M
	# Step 5: Check if the filename ends with '.npy', indicating a NumPy binary file.
	elif filename.endswith('.npy'):
		# Step 6: Load the data from the '.npy' file directly into a NumPy array.
		M = np.load(filename)
		print('\tOriginal data shape:', M.shape)
		return M
	# If the file format is not supported, print an error message and return None.
	else:
		print('\tERROR: unsupported format for input file.')
		return None

filename1 = 'time_dSOAP_0k_20k.npz'
filename2 = 'time_dSOAP_20k_40k.npz'
M1 = read_data('data/' + filename1)
M2 = read_data('data/' + filename2)

M = np.array([ np.concatenate((M1[i], M2[i])) for i in range(len(M1)) ])

np.save('data/time_dSOAP_freezing_0k_40k.npy', M)