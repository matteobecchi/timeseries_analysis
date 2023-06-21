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
from matplotlib import cm
import hdbscan
import plotly
plotly.__version__
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import seaborn as sns

def read_input_parameters():
	filename = np.loadtxt('data_directory.txt', dtype=str)
	param = np.loadtxt('input_parameters.txt')
	tau_window = int(param[0])
	tau_smooth = int(param[1])
	tau_delay = int(param[2])
	number_of_sigmas = param[3]
	if filename.shape == (2,):
		return filename, tau_window, tau_smooth, tau_delay, number_of_sigmas
	else:
		return str(filename), tau_window, tau_smooth, tau_delay, number_of_sigmas

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

def relabel_states(all_the_labels, list_of_states):
	### Remove empty states
	list1 = [ state for state in list_of_states if state[2] != 0.0 ]
	list_unique = np.unique(all_the_labels)
	tmp = all_the_labels
	for i, l in enumerate(list_unique):
		for a in range(len(all_the_labels)):
			for b in range(len(all_the_labels[a])):
				if all_the_labels[a][b] == l:
					tmp[a][b] = i

	### Swappino
	list2 = list1[1:]
	list2.append(list1[0])
	for a in range(len(tmp)):
		for b in range(len(tmp[a])):
				tmp[a][b] = (tmp[a][b] - 1)%list_unique.size

	return tmp, list2

def Sankey(all_the_labels, t_start, number_of_frames, filename):
	print('* Computing and plotting the averaged Sankey diagrams...')
	if t_start + number_of_frames > all_the_labels.shape[1]:
		print('ERROR: the required time range is out of bound.')
		return

	n_states = np.unique(all_the_labels).size
	T = np.zeros((n_states, n_states))
	for t in range(t_start, t_start + number_of_frames):
		for L in all_the_labels:
			T[int(L[t])][int(L[t + 1])] += 1

	source = np.empty(n_states**2)
	target = np.empty(n_states**2)
	value = np.empty(n_states**2)
	c = 0
	for n1 in range(len(T)):
		for n2 in range(len(T[n1])):
			source[c] = n1
			target[c] = n2 + n_states
			value[c] = T[n1][n2]
			c += 1

	# label = np.tile(range(n_states), n_steps)
	label = np.tile(['Ice', 'Liquid', 'Interface', 'State_3', 'Fast_Dyn'], 2)

	palette = sns.color_palette('viridis', n_colors=n_states).as_hex()
	color = np.tile(palette, 2)

	node = dict(label=label, pad=30, thickness=20, color=color)
	link = dict(source=source, target=target, value=value)

	Data = go.Sankey(link=link, node=node, arrangement="perpendicular")
	fig = go.Figure(Data)

	fig.show()
	fig.write_image(filename + '.png', scale=5.0)

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
