import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from matplotlib.pyplot import imshow
from matplotlib.colors import LogNorm
from matplotlib import cm
import plotly
plotly.__version__
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import normalize

def read_input_parameters():
	filename = np.loadtxt('data_directory.txt', dtype=str)
	param = np.loadtxt('input_parameters.txt')
	tau_window = int(param[0])
	tau_delay = int(param[1])
	t_conv = param[2]
	tau_smooth = int(param[3])
	number_of_sigmas = param[4]
	example_ID = int(param[5])
	PAR = [tau_window, tau_delay, t_conv, tau_smooth, number_of_sigmas, example_ID]
	if filename.shape == (2,):
		return filename, PAR
	else:
		return str(filename), PAR

def read_data(filename):
	print('* Reading data...')
	if filename[-3:] == 'npz':
		with np.load(filename) as data:
			lst = data.files
			M = np.array(data[lst[0]])
			if M.ndim == 3:
				M = np.vstack(M)
				M = M.T
				print('\tData shape:', M.shape)
			return M
	elif filename[-3:] == 'npy':
		M = np.load(filename)
		print('\tData shape:', M.shape)
		return M
	else:
		print('Error: unsupported format for input file.')
		return

def normalize_array(x):
	mean = np.mean(x)
	stddev = np.sqrt(np.var(x))
	tmp = (x - mean)/stddev
	return tmp, mean, stddev

def plot_histo(ax, counts, bins):
	ax.stairs(counts, bins, fill=True)
	ax.set_xlabel(r'Normalized signal')
	ax.set_ylabel(r'Probability distribution')

def Savgol_filter(M, tau, poly_order):
	return np.array([ savgol_filter(x, tau, poly_order) for x in M ])

def gaussian(x, m, sigma, A):
	return A*np.exp(-((x - m)/sigma)**2)

def exponential(x, A, nu):
	return A*np.exp(-x*nu)

def remove_first_points(M, delay):
	### to remove the first delay frames #####
	tmp = []
	for m in M:
		tmp.append(m[delay:])
	return np.array(tmp)

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def relabel_states(all_the_labels, list_of_states):
	### Remove empty states and relabel from 0 to n_states-1
	list1 = [ state for state in list_of_states if state[2] != 0.0 ]
	list_unique = np.unique(all_the_labels)
	tmp1 = all_the_labels
	for i, l in enumerate(list_unique):
		for a in range(len(all_the_labels)):
			for b in range(len(all_the_labels[a])):
				if all_the_labels[a][b] == l:
					tmp1[a][b] = i

	### Swappino (I want the Dynamic state to be the last one)
	for a in range(len(tmp1)):
		for b in range(len(tmp1[a])):
				tmp1[a][b] = (tmp1[a][b] - 1)%list_unique.size

	### Order the states according to the mu values
	# list_of_mu = np.array([ state[0][0] for state in list1 ])
	# copy_of_list_of_mu = np.array([ state[0][0] for state in list1 ])
	# sorted_IDs = []
	# while len(copy_of_list_of_mu) > 0:
	# 	min_mu = np.min(copy_of_list_of_mu)
	# 	ID_min = np.where(list_of_mu == min_mu)[0][0]
	# 	sorted_IDs.append(ID_min)
	# 	ID_min2 = np.where(copy_of_list_of_mu == min_mu)[0][0]
	# 	copy_of_list_of_mu = np.delete(copy_of_list_of_mu, ID_min2)
	# list2 = []
	# for ID in sorted_IDs:
	# 	list2.append(list1[ID])

	# tmp2 = copy.deepcopy(tmp1)
	# for i, l in enumerate(sorted_IDs):
	# 	for a in range(len(tmp1)):
	# 		for b in range(len(tmp1[a])):
	# 			if tmp1[a][b] == l:
	# 				tmp2[a][b] = i

	tmp2 = tmp1
	list2 = list1

	# with open('tmp_state_data.txt', 'w') as f:
	# 	for state in list2:
	# 		print(state[2], file=f)

	return tmp2, list2

def print_mol_labels1(all_the_labels, PAR, filename):
	tau_window = PAR[0]
	with open(filename, 'w') as f:
		for i in range(all_the_labels.shape[0]):
			string = str(all_the_labels[i][0])
			for t in range(1, tau_window):
					string += ' ' + str(all_the_labels[i][0])
			for w in range(1, all_the_labels[i].size):
				for t in range(tau_window):
					string += ' ' + str(all_the_labels[i][w])
			print(string, file=f)

