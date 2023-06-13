import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def read_data(filename):
	print('* Reading data...')
	with np.load(filename) as data:
		lst = data.files
		M = np.array(data[lst[0]])
		print('\tData shape:', M.T.shape)
		if M.ndim == 3:
			M = np.vstack(M)
			M = M.T
			print('\tData shape after reshaping:', M.shape)
		return M

def plot_histo(ax, counts, bins):
	ax.stairs(counts, bins, fill=True)
	ax.set_xlabel(r'$t$SOAP signal')
	ax.set_ylabel(r'Probability distribution')

def Savgol_filter(M, tau, poly_order):
	return np.array([ savgol_filter(x, tau, poly_order) for x in M ])

def gaussian(x, m, sigma, A):
	return A*np.exp(-((x - m)/sigma)**2)

def remove_first_points(M, delay):
	### to remove the first delay frames #####
	tmp = []
	for m in M:
		tmp.append(m[delay:])
	return tmp

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def plot_and_save_histogram(M, n_bins, filename):
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	fig, ax = plt.subplots()
	plot_histo(ax, counts, bins)
	plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def print_mol_labels1(all_the_labels, tau_window, filename):
	with open(filename, 'w') as f:
		for i in range(all_the_labels.shape[0]):
			string = str(all_the_labels[i][0])
			for t in range(1, tau_window):
					string += ' ' + str(all_the_labels[i][0])
			for w in range(1, all_the_labels[i].size):
				for t in range(tau_window):
					string += ' ' + str(all_the_labels[i][w])
			print(string, file=f)

def print_mol_labels2(all_the_labels, tau_window, filename):
	with open(filename, 'w') as f:
		for w in range(all_the_labels.shape[1]):
			for t in range(tau_window):
				for i in range(all_the_labels.shape[0]):
					print(all_the_labels[i][w], file=f)

def read_input_parameters():
	filename = np.loadtxt('data_directory.txt', dtype=str)
	if filename.shape == (2,):
		return filename
	else:
		return str(filename)
