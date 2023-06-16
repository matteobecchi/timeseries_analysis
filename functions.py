import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from matplotlib.pyplot import imshow
from matplotlib.colors import LogNorm
import hdbscan

def read_data(filename):
	print('* Reading data...')
	with np.load(filename) as data:
		lst = data.files
		M = np.array(data[lst[0]])
		if M.ndim == 3:
			M = np.vstack(M)
			M = M.T
			print('\tData shape:', M.shape)
		return M

def normalize_array(x):
	mean = np.mean(x)
	stddev = np.sqrt(np.var(x))
	tmp = (x - mean)/stddev
	return tmp, mean, stddev

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

def plot_and_save_histogram(M, n_bins, tSOAP_lim, filename):
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	fig, ax = plt.subplots()
	plot_histo(ax, counts, bins)
	ax.set_xlim(tSOAP_lim)
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
	param = np.loadtxt('input_parameters.txt')
	tau_smooth = int(param[0])
	tau_delay = int(param[1])
	number_of_sigmas = param[2]
	if filename.shape == (2,):
		return filename, tau_smooth, tau_delay, number_of_sigmas
	else:
		return str(filename), tau_smooth, tau_delay, number_of_sigmas

def normalize_T_matrix(T):
	N = np.empty(T.shape)
	for i in range(len(T)):
		S = np.sum(T[i])
		if S != 0.0:
			for j in range(len(T[i])):
				N[i][j] = T[i][j]/S
	return N

def compute_transition_matrix(all_the_labels, filename):
	unique_labels = np.unique(all_the_labels)
	n_states = unique_labels.size
	
	def rename_index(ID, unique_labels):
		for i, l in enumerate(unique_labels):
			if ID == l:
				return i
	
	T = np.zeros((n_states, n_states))
	for lablist in all_the_labels:
		for w in range(len(lablist) - 1):
			ID0 = rename_index(lablist[w], unique_labels)
			ID1 = rename_index(lablist[w + 1], unique_labels)
			T[ID0][ID1] += 1
	number_of_transitions = np.sum(T)
	T /= number_of_transitions
	# print(T)

	fig, ax = plt.subplots(figsize=(10, 8))
	im = ax.imshow(T, norm=LogNorm(vmin=0.000001, vmax=1))
	fig.colorbar(im)
	for (i, j),val in np.ndenumerate(T):
		ax.text(j, i, "{:.2f}".format(100*val), ha='center', va='center')
	ax.set_xlabel('To...')
	ax.set_ylabel('From...')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	plt.show()
	fig.savefig(filename + '.png', dpi=600)

	return T

def HDBSCAN_clustering(data):
	norm_x, mux, sigmax = normalize_array(data.T[0])
	norm_y, muy, sigmay = normalize_array(data.T[1])
	data = np.array([norm_x, norm_y]).T

	clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=40,
		metric='euclidean', min_cluster_size=50, min_samples=3 , p=None).fit(data)

	labels = clusterer.labels_

	fig, ax = plt.subplots(figsize=(7.5, 4.8))
	for n in np.unique(labels):
		ax.scatter(data[labels == n, 0]*sigmax + mux, data[labels == n, 1]*sigmay + muy, s=2)
	plt.show()

