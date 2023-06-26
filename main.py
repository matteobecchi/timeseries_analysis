import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import pycwt as wavelet
from functions import *

### System specific parameters ###
t_units = r'[ns]'			# Units of measure of time
y_units = r'[$t$SOAP]'		# Units of measure of the signal
example_ID = 100

### Usually no need to changhe these ###
output_file = 'states_output.txt'
poly_order = 2 				# Savgol filter polynomial order
n_bins = 100 				# Number of bins in the histograms
stop_th = 0.01 				# Treshold to exit the maxima search
show_plot = True			# Show all the plots

def all_the_input_stuff():
	### Read and clean the data points
	data_directory, PAR = read_input_parameters()
	### Create file for output
	with open(output_file, 'w') as f:
		print('# ' + str(PAR[0]) + ', ' + str(PAR[2]) + ', ' + str(PAR[3]), file=f)
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])
	M_raw = remove_first_points(M_raw, PAR[2])
	M = Savgol_filter(M_raw, PAR[1], poly_order)
	SIG_MAX = np.max(M)
	SIG_MIN = np.min(M)
	M = (M - SIG_MIN)/(SIG_MAX - SIG_MIN)
	total_time = M.shape[1]
	print('* Using ' + str(int(total_time/PAR[0])) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[4]) + ' ns). ')
	all_the_labels = np.zeros((len(M), int(total_time/PAR[0])))
	list_of_states = []

	return M_raw, M, PAR, all_the_labels, list_of_states

def gauss_fit_n(M, n_bins, number_of_sigmas, filename):
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)

	### Locate the maxima and minima in the distribution
	max_ID = argrelextrema(counts, np.greater)[0]
	min_ID = argrelextrema(counts, np.less_equal)[0]
	tmp_to_delete = []
	for n in range(min_ID.size - 1):
		if min_ID[n + 1] == min_ID[n] + 1:
			tmp_to_delete.append(n + 1)
	min_ID = np.delete(min_ID, tmp_to_delete, 0)
	tmp_min_ID = [min_ID[0]]
	current_max = 0
	for n in range(1, len(min_ID) - 1):
		if min_ID[n] > max_ID[current_max]:
			tmp_min_ID.append(min_ID[n])
			current_max += 1
	tmp_min_ID.append(min_ID[-1])

	min_ID = np.array(tmp_min_ID)

	list_popt = []
	for n in range(max_ID.size):
		### Chose the intersection interval between
		### the width at half height and the minima surrounding the maximum
		counts_max = counts[max_ID[n]]
		tmp_id0 = max_ID[n]
		tmp_id1 = max_ID[n]
		while (counts[tmp_id0] > counts_max/2 and tmp_id0 > 0):
			tmp_id0 -= 1
		while (counts[tmp_id1] > counts_max/2 and tmp_id1 < len(counts) - 1):
			tmp_id1 += 1
		id0 = np.max([tmp_id0, min_ID[n]])
		id1 = np.min([tmp_id1, min_ID[n + 1]])
		if id1 - id0 < 4: # If the fitting interval is too small, discard.
			continue
		Bins = bins[id0:id1]
		Counts = counts[id0:id1]

		### Perform the Gaussian fit
		p0 = [bins[max_ID[n]], (bins[min_ID[n + 1]] - bins[min_ID[n]])/6, counts[max_ID[n]]]
		try:
			popt, pcov = scipy.optimize.curve_fit(gaussian, Bins, Counts, p0=p0)
		except RuntimeError:
			print('gauss_fit_n: RuntimeError.')
			continue
		popt[2] *= flat_M.size
		if popt[1] < 0:
			popt[1] = -popt[1]
		flag = 1
		if popt[0] < Bins[0] or popt[0] > Bins[-1]:
			flag = 0 # If mu is outside the fitting range, it's not identifying the right Gaussian. Discard. 
		if popt[1] > Bins[-1] - Bins[0]:
			flag = 0 # If sigma is larger than the fitting interval, it's not identifying the right Gaussian. Discard. 
		perr = np.sqrt(np.diag(pcov))
		for j in range(len(perr)):
			if perr[j]/popt[j] > 0.5:
				flag = 0 # If the uncertanties over the parameters is too large, discard.
		if flag:
			list_popt.append(popt)

	print('* Gaussians parameters:')
	with open(output_file, 'a') as f:
		print('\n', file=f)
		for popt in list_popt:
			print(f'\tmu = {popt[0]:.4f}, sigma = {popt[1]:.4f}, amplitude = {popt[2]:.4f}')
			print(f'\tmu = {popt[0]:.4f}, sigma = {popt[1]:.4f}, amplitude = {popt[2]:.4f}', file=f)

	### Create the list of the trasholds for state identification
	list_th = []
	for n in range(len(list_popt)):
		th_inf = list_popt[n][0] - number_of_sigmas*list_popt[n][1]
		th_sup = list_popt[n][0] + number_of_sigmas*list_popt[n][1]
		list_th.append([th_inf, th_sup])
	
	### To remove possible swapped tresholds:
	for n in range(len(list_th) - 1):
		if list_th[n][1] > list_th[n + 1][0]:
			mu0 = list_popt[n][0]
			sigma0 = list_popt[n][1]
			mu1 = list_popt[n + 1][0]
			sigma1 = list_popt[n + 1][1]
			middle_th = (mu0/sigma0 + mu1/sigma1)/(1/sigma0 + 1/sigma1)
			list_th[n][1] = middle_th
			list_th[n + 1][0] = middle_th

	### Plot the distribution and the fitted Gaussians
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	fig, ax = plt.subplots()
	plot_histo(ax, counts, bins)
	ax.set_xlim(y_lim)
	for popt in list_popt:
		tmp_popt = [popt[0], popt[1], popt[2]/flat_M.size]
		ax.plot(np.linspace(bins[0], bins[-1], 1000), gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))
	for th in list_th:
		ax.vlines(th, 0, np.max(counts), linestyle='--', color='black')
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return list_popt, list_th

def find_stable_trj(M, tau_window, list_th, list_of_states, all_the_labels, offset):
	number_of_windows = int(M.shape[1]/tau_window)
	M2 = []
	counter = [ 0 for n in range(len(list_th)) ]
	for i, x in enumerate(M):
		for w in range(number_of_windows):
			if all_the_labels[i][w] > 0.5:
				continue
			else:
				x_w = x[w*tau_window:(w + 1)*tau_window]
				flag = 1
				for l, th in enumerate(list_th):
					if np.amin(x_w) > th[0] and np.amax(x_w) < th[1]:
						all_the_labels[i][w] = l + offset + 1
						counter[l] += 1
						flag = 0
						break
				if flag:
					M2.append(x_w)

	print('* Finding stable windows...')
	with open(output_file, 'a') as f:
		for n, c in enumerate(counter):
			fw = c/(all_the_labels.size)
			print(f'\tFraction of windows in state ' + str(offset + n + 1) + f' = {fw:.3}')
			print(f'\tFraction of windows in state ' + str(offset + n + 1) + f' = {fw:.3}', file=f)
			list_of_states[len(list_of_states) - len(counter) + n][2] = fw
	return np.array(M2), np.sum(counter)/(len(M)*number_of_windows), list_of_states

def iterative_search(M, PAR, all_the_labels, list_of_states):
	M1 = M
	iteration_id = 1
	states_counter = 0
	while True:
		### Locate and fit maxima in the signal distribution
		list_popt, list_th = gauss_fit_n(M1, n_bins, PAR[3], 'output_figures/Fig1_' + str(iteration_id))

		for n in range(len(list_th)):
			list_of_states.append([list_popt[n], list_th[n], 0.0])

		### Find the windows in which the trajectories are stable in one maxima
		M2, c, list_of_states = find_stable_trj(M, PAR[0], list_th, list_of_states, all_the_labels, states_counter)

		states_counter += len(list_popt)
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c < stop_th:
			break
		else:
			M1 = M2

	return relabel_states(all_the_labels, list_of_states)

def plot_all_trajectories(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing colored trajectories with histograms...')
	tau_window = PAR[0]
	tau_delay = PAR[2]
	t_conv = PAR[4]
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	States = np.unique(all_the_labels)
	for c, S in enumerate(States):
		list_of_times = []
		list_of_signals = []
		for i, L in enumerate(all_the_labels):
			if i%10!=0:
				continue
			for w, l in enumerate(L):
				if l == S:
					t0 = w*tau_window
					t1 = (w + 1)*tau_window
					list_of_times.append(np.linspace((tau_delay + t0)*t_conv, (tau_delay + t1)*t_conv, tau_window))
					list_of_signals.append(M[i][t0:t1])
		list_of_times = np.array(list_of_times)
		list_of_signals = np.array(list_of_signals)
		flat_times = list_of_times.flatten()
		flat_signals = list_of_signals.flatten()
		flat_colors = c*np.ones(flat_times.size)

		fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
		fig.suptitle('State ' + str(c))
		ax[0].scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States), s=0.05, alpha=0.5, rasterized=True)
		ax[0].set_xlabel(r'Time ' + t_units)
		ax[0].set_ylabel(r'Signal ' + y_units)
		y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
		ax[0].set_ylim(y_lim)
		ax[1].stairs(counts, bins, fill=True, orientation='horizontal')
		if c < len(States) - 1:
			ax[1].hlines(list_of_states[c][1], xmin=0.0, xmax=np.amax(counts), linestyle='--', color='black')
			ax[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *list_of_states[c][0]), np.linspace(bins[0], bins[-1], 1000))
		if show_plot:
			plt.show()
		fig.savefig(filename + str(c) + '.png', dpi=600)
		plt.close(fig)

def plot_one_trajectory(x, PAR, L, list_of_states, States, y_lim, filename):
	tau_window = PAR[0]
	tau_delay = PAR[2]
	t_conv = PAR[4]
	fig, ax = plt.subplots()
	for c, S in enumerate(States):
		list_of_times = []
		list_of_signals = []
		for w, l in enumerate(L):
			if l == S:
				t0 = w*tau_window
				t1 = (w + 1)*tau_window
				list_of_times.append(np.linspace((tau_delay + t0)*t_conv, (tau_delay + t1)*t_conv, tau_window))
				list_of_signals.append(x[t0:t1])
		list_of_times = np.array(list_of_times)
		list_of_signals = np.array(list_of_signals)
		flat_times = list_of_times.flatten()
		flat_signals = list_of_signals.flatten()
		flat_colors = c*np.ones(flat_times.size)

		ax.scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States))
	
	ax.set_xlabel(r'Time ' + t_units)
	ax.set_ylabel(r'Signal ' + y_units)
	ax.set_ylim(y_lim)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def state_statistics(M, PAR, all_the_labels, resolution, filename):
	print('* Computing some statistics on the states...')
	tau_window = PAR[0]
	t_conv = PAR[4]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	data = []
	labels = []
	for i, x in enumerate(M):
		current_label = all_the_labels[i][0]
		x_w = x[0:tau_window]
		for w in range(1, number_of_windows):
			if all_the_labels[i][w] == current_label:
				x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
			else:
				if x_w.size < tau_window*resolution:
			 		continue
				data.append([x_w.size*t_conv, np.mean(x_w)])
				labels.append(current_label)
				x_w = x[tau_window*w:tau_window*(w + 1)]
				current_label = all_the_labels[i][w]

	data = np.array(data).T
	fig, ax = plt.subplots()
	ax.scatter(data[0], data[1], c=labels, s=1.0)
	ax.set_xlabel(r'State duration $T$ ' + t_units)
	ax.set_ylabel(r'State mean amplitude ' + y_units)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def main():
	M_raw, M, PAR, all_the_labels, list_of_states = all_the_input_stuff()

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)

	plot_all_trajectories(M, PAR, all_the_labels, list_of_states, 'output_figures/Fig2_')
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	plot_one_trajectory(M[example_ID], PAR, all_the_labels[example_ID], list_of_states, np.unique(all_the_labels), y_lim, 'output_figures/Fig3')

	state_statistics(M, PAR, all_the_labels, 1, 'output_figures/Fig4')

	t_start = 0
	for t_jump in [1, 10]:
		Sankey(all_the_labels, t_start, t_jump, 9, PAR[4], 'output_figures/Fig5_' + str(t_start) + '-' + str(t_jump))
	compute_transition_matrix(PAR, all_the_labels, 'output_figures/Fig6')

	print_mol_labels1(all_the_labels, PAR, 'all_cluster_IDs.dat')

if __name__ == "__main__":
	main()
