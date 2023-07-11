import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import sys
import os
import copy
import math
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
    data_dir = np.loadtxt('data_directory.txt', dtype=str)

    with open('input_parameters.txt', 'r') as file:
        lines = file.readlines()
        param = [float(line.strip()) for line in lines]

    tau_window = int(param[0])
    tau_delay = int(param[1])
    t_conv = param[2]
    example_ID = int(param[3])
    PAR = [tau_window, tau_delay, t_conv, example_ID]

    if data_dir.shape == (2, ):
        return data_dir, PAR
    else:
        return str(data_dir), PAR

def read_data(filename):
	print('* Reading data...')
	if filename.endswith('.npz'):
		with np.load(filename) as data:
			lst = data.files
			M = np.array(data[lst[0]])
			if M.ndim == 3:
				M = np.vstack(M)
				M = M.T
			if M.shape[0] != 2048:
				M = M.T
			print('\tData shape:', M.shape)
			return M
	elif filename.endswith('.npy'):
		M = np.load(filename)
		print('\tData shape:', M.shape)
		return M
	else:
		print('Error: unsupported format for input file.')
		return None

def Savgol_filter(M, tau, poly_order):
	tmp = np.array([ savgol_filter(x, tau, poly_order) for x in M ])
	return tmp[:, int(tau/2):-int(tau/2)]

def normalize_array(x):
	mean = np.mean(x)
	stddev = np.std(x)
	tmp = (x - mean)/stddev
	return tmp, mean, stddev

def plot_histo(ax, counts, bins):
	ax.stairs(counts, bins, fill=True)
	ax.set_xlabel(r'Normalized signal')
	ax.set_ylabel(r'Probability distribution')

def gaussian(x, m, sigma, A):
	return A*np.exp(-((x - m)/sigma)**2)

def exponential(t, tau):
	return np.exp(-t/tau)/tau

def double_exp(t, tau1, tau2):
	return np.exp(-t/tau1)/tau1 + np.exp(-t/tau2)/tau2

def cumulative_exp(t, tau):
	return 1 - np.exp(-t/tau)

def cumulative_double_exp(t, tau1, tau2):
	return 1 - tau1*np.exp(-t/tau1)/(tau1 + tau2) - tau2*np.exp(-t/tau2)/(tau1 + tau2)

def find_nearest(array, value):
	array = np.asarray(array)
	idx = np.argmin(np.abs(array - value))
	return array[idx]

def relabel_states(all_the_labels, list_of_states, stop_th):
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
	list_of_mu = np.array([ state[0][0] for state in list1 ])
	copy_of_list_of_mu = np.array([ state[0][0] for state in list1 ])
	sorted_IDs = []
	while len(copy_of_list_of_mu) > 0:
		min_mu = np.min(copy_of_list_of_mu)
		ID_min = np.where(list_of_mu == min_mu)[0][0]
		sorted_IDs.append(ID_min)
		ID_min2 = np.where(copy_of_list_of_mu == min_mu)[0][0]
		copy_of_list_of_mu = np.delete(copy_of_list_of_mu, ID_min2)
	list2 = []
	for ID in sorted_IDs:
		list2.append(list1[ID])

	tmp2 = copy.deepcopy(tmp1)
	for i, l in enumerate(sorted_IDs):
		for a in range(len(tmp1)):
			for b in range(len(tmp1[a])):
				if tmp1[a][b] == l:
					tmp2[a][b] = i

	return tmp2, list2

def set_final_states(list_of_states):
	final_list = []
	final_list.append([0.0, 0])
	for s in range(len(list_of_states) - 1):
		mu0 = list_of_states[s][0][0]
		mu1 = list_of_states[s + 1][0][0]
		sigma0 = list_of_states[s][0][1]
		sigma1 = list_of_states[s + 1][0][1]
		A0 = list_of_states[s][0][2]
		A1= list_of_states[s + 1][0][2]
		a = sigma1**2 - sigma0**2
		b = -2*(mu0*sigma1**2 - mu1*sigma0**2)
		c = (mu0*sigma1)**2 - (mu1*sigma0)**2 - (sigma0*sigma1)**2*np.log(A0/A1)
		Delta = b**2 - 4*a*c
		if Delta >= 0:
			th_plus = (- b + np.sqrt(Delta))/(2*a)
			th_minus = (- b - np.sqrt(Delta))/(2*a)
			intercept_plus = gaussian(th_plus, mu0, sigma0, A0)
			intercept_minus = gaussian(th_minus, mu0, sigma0, A0)
			if intercept_plus >= intercept_minus:
				final_list.append([th_plus, 1])
			else:
				final_list.append([th_minus, 1])
		else:
			final_list.append([(mu0/sigma0 + mu1/sigma1)/(1/sigma0 + 1/sigma1), 2])
	final_list.append([1.0, 0])

	return final_list

def assign_final_states(M, PAR, final_list):
	print('* Assigning labels to the time windows...')
	tau_window = PAR[0]
	number_of_windows = int(M.shape[1]/tau_window)
	all_the_labels = np.empty((M.shape[0], number_of_windows))
	for i in range(M.shape[0]):
		for w in range(number_of_windows):
			x_w = M[i][w*tau_window:(w + 1)*tau_window]
			flag = 0
			for l in range(len(final_list) - 1):
				if np.min(x_w) > final_list[l] and np.max(x_w) < final_list[l + 1]:
					all_the_labels[i][w] = l
					flag = 1
			if flag == 0:
				all_the_labels[i][w] = len(final_list) - 1

	return all_the_labels

def assign_final_states_to_single_frames(M, PAR, final_list):
	print('* Assigning labels to all the single frames...')
	all_the_labels = np.empty((M.shape[0], M.shape[1]))
	for i in range(M.shape[0]):
		for t in range(M.shape[1]):
			flag = 0
			for l in range(len(final_list) - 1):
				if M[i][t] >= final_list[l][0] and M[i][t] <= final_list[l + 1][0]:
					all_the_labels[i][t] = l
					flag = 1
			if flag == 0:
				all_the_labels[i][t] = len(final_list) - 1

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

def print_mol_labels_fbf(all_the_labels, PAR, filename):
	print('* Print color IDs for Ovito...')
	with open(filename, 'w') as f:
		for i in range(all_the_labels.shape[0]):
			string = str(all_the_labels[i][0])
			for t in range(1, all_the_labels[i].size):
				string += ' ' + str(all_the_labels[i][t])
			print(string, file=f)

def print_mol_labels_fbf2(all_the_labels, PAR, filename):
	print(all_the_labels.shape)
	print('* Print color IDs for Ovito...')
	with open(filename, 'w') as f:
		for t in range(481):
			print('#', file=f)
			print('#', file=f)
			for i in range(all_the_labels.shape[0]):
				print(all_the_labels[i][t], file=f)

def transition_matrix(all_the_labels, filename, show_plot):
	print('* Computing transition matrix...')
	n_states = np.unique(all_the_labels).size
	T = np.zeros((n_states, n_states))

	for mol in all_the_labels:
		for t in range(mol.size - 1):
			id0 = int(mol[t])
			id1 = int(mol[t + 1])
			T[id0][id1] += 1.0

	N = np.zeros((n_states, n_states))
	for i, row in enumerate(T):
		if np.sum(row) > 0:
			for j, el in enumerate(row):
				N[i][j] = row[j]/np.sum(row)

	N_min = np.max(N)
	for (i, j), val in np.ndenumerate(N):
		if val < N_min and val > 0.0:
			N_min = val

	fig, ax = plt.subplots(figsize=(10, 8))
	im = ax.imshow(N, cmap='viridis', norm=LogNorm(vmin=N_min, vmax=np.max(N)))
	fig.colorbar(im)
	for (i, j), val in np.ndenumerate(N):
		ax.text(j, i, "{:.2f}".format(val), ha='center', va='center')
	fig.suptitle(r'Transition probabilities')
	ax.set_xlabel('To...')
	ax.set_ylabel('From...')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_xticks(np.linspace(0.0, n_states - 1.0, n_states))
	ax.set_xticklabels(range(n_states))
	ax.set_yticks(np.linspace(0.0, n_states - 1.0, n_states))
	ax.set_yticklabels(range(n_states))
	
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)




