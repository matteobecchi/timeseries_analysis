import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
import pycwt as wavelet
from functions import *

### Other stuff, usually no need to changhe these ###
output_file = 'states_output.txt'
poly_order = 2 				# Savgol filter polynomial order
n_bins = 100 				# Number of bins in the histograms
stop_th = 0.01 				# Treshold to exit the maxima search
t_units = r'[ns]'			# Units of measure of time
t_conv = 0.001 				# Conversion between frames and time units
y_units = r'[$t$SOAP]'		# Units of measure of the signal
tSOAP_lim = [0.014, 0.044]	# Limit of the x axes for the histograms
replot = False				# Plot all the data distribution during the maxima search

def all_the_input_stuff():
	### Read and clean the data points
	data_directory, tau_window, tau_smooth, tau_delay, number_of_sigmas = read_input_parameters()
	### Create file for output
	with open(output_file, 'w') as f:
		print('# ' + str(tau_window) + ', ' + str(tau_delay) + ', ' + str(number_of_sigmas), file=f)
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])
	M_raw = remove_first_points(M_raw, tau_delay)
	M = Savgol_filter(M_raw, tau_smooth, poly_order)
	return M_raw, M, tau_window, tau_delay, number_of_sigmas

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

	### Some tries and checks to identify and discard fake maxima
	list_popt = []
	for n in range(max_ID.size):
		if min_ID[n + 1] - min_ID[n] < 5:
			continue
		B = bins[min_ID[n]:min_ID[n + 1]]
		C = counts[min_ID[n]:min_ID[n + 1]]
		try:
			popt, pcov = scipy.optimize.curve_fit(gaussian, B, C)
		except RuntimeError:
			print('gauss_fit_n: RuntimeError.')
			continue
		popt[2] *= flat_M.size
		if popt[1] < 0:
			popt[1] = -popt[1]
		flag = 1
		if popt[0] < B[0] or popt[0] > B[-1]:
			flag = 0
		perr = np.sqrt(np.diag(pcov))
		for j in range(len(perr)):
			if perr[j]/popt[j] > 0.5:
				flag = 0
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

	if replot:
		### Plot the distribution and the fitted Gaussians
		fig, ax = plt.subplots()
		plot_histo(ax, counts, bins)
		ax.set_xlim(tSOAP_lim)
		for popt in list_popt:
			ax.plot(np.linspace(bins[0], bins[-1], 1000), gaussian(np.linspace(bins[0], bins[-1], 1000), *popt))
		for th in list_th:
			ax.vlines(th, 0, 100, linestyle='--', color='black')
		ax.set_xlim(tSOAP_lim)
		plt.show()
		fig.savefig(filename + '.png', dpi=600)

	return list_popt, list_th

def find_stable_trj(M, list_th, list_of_states, number_of_windows, tau_window, all_the_labels, offset):
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

def plot_trajectories_after(M, all_the_labels, list_of_states, tau_window, tau_delay):
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	States = np.unique(all_the_labels)
	for c, S in enumerate(States):
		list_of_times = []
		list_of_signals = []
		for i, L in enumerate(all_the_labels):
			if i%50!=0:
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
		ax[0].scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States), s=0.1, rasterized=True)
		ax[0].set_xlabel(r'Time ' + t_units)
		ax[0].set_ylabel(r'$t$SOAP signal ' + y_units)
		ax[0].set_ylim(tSOAP_lim)
		ax[1].stairs(counts, bins, fill=True, orientation='horizontal')
		if c > 0:
			ax[1].hlines(list_of_states[c - 1][1], xmin=0.0, xmax=np.amax(counts), linestyle='--', color='black')
			ax[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *list_of_states[c - 1][0]), np.linspace(bins[0], bins[-1], 1000))
		plt.show()

def tau_sigma(M, all_the_labels, number_of_windows, tau_window, resolution, filename):
	data = []
	labels = []
	wc = 0
	for i, x in enumerate(M):
		current_label = all_the_labels[i][0]
		x_w = x[0:tau_window]
		for w in range(1, number_of_windows):
			 if all_the_labels[i][w] == current_label:
			 	x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
			 else:
			 	if x_w.size < tau_window*resolution:
			 		continue
			 	### Lag-1 autocorrelation for colored noise
				### fitting with x_n = \alpha*x_{n-1} + z_n, z_n gaussian white noise
			 	x_smean = x_w - np.mean(x_w)
			 	alpha = 0.0
			 	var = 1.0
			 	try:
			 		alpha, var, _ = wavelet.ar1(x_smean)
			 		data.append([alpha, np.sqrt(var)])
			 		labels.append(current_label)
			 	except Warning:
			 		wc += 1
			 	x_w = x[tau_window*w:tau_window*(w + 1)]
			 	current_label = all_the_labels[i][w]

	print('\t-- ' + str(wc) + ' warnings generated --')
	data = np.array(data).T
	tau_c = 1/(1 - data[0])

	figa, axa = plt.subplots(figsize=(7.5, 4.8))
	axa.scatter(tau_c, data[1], c='xkcd:black', s=1.0)
	axa.set_xlabel(r'Correlation time $\tau_c$ [ps]')
	axa.set_ylabel(r'Gaussian noise amplitude $\sigma_n$ ' + y_units)
	figa.savefig(filename + 'a.png', dpi=600)

	figb, axb = plt.subplots(figsize=(7.5, 4.8))
	axb.scatter(tau_c, data[1], c=labels, s=1.0)
	axb.set_xlabel(r'Correlation time $\tau_c$ [ps]')
	axb.set_ylabel(r'Gaussian noise amplitude $\sigma_n$ ' + y_units)
	# axb.legend()
	figb.savefig(filename + 'b.png', dpi=600)
	
	plt.show()

def main():
	M_raw, M, tau_window, tau_delay, number_of_sigmas = all_the_input_stuff()
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	print('* Using ' + str(number_of_windows) + ' windows of length ' + str(tau_window) + ' frames (' + str(tau_window*t_conv) + ' ns). ')
	all_the_labels = np.zeros((len(M), number_of_windows))
	list_of_states = []

	M1 = M
	iteration_id = 1
	states_counter = 0
	while True:
		### Locate and fit maxima in the signal distribution
		list_popt, list_th = gauss_fit_n(M1, n_bins, number_of_sigmas, 'output_figures/Fig' + str(iteration_id))

		for n in range(len(list_th)):
			list_of_states.append([list_popt[n], list_th[n], 0.0])

		### Find the windows in which the trajectories are stable in one maxima
		M2, c, list_of_states = find_stable_trj(M, list_th, list_of_states, number_of_windows, tau_window, all_the_labels, states_counter)

		states_counter += len(list_popt)
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c < stop_th:
			break
		else:
			M1 = M2

	all_the_labels, list_of_states = relabel_states(all_the_labels, list_of_states)

	plot_trajectories_after(M, all_the_labels, list_of_states, tau_window, tau_delay)

	### Amplitude vs time of the windows scatter plot
	# print('* Computing the amplitude - correlation diagram...')
	# for i, tmin in enumerate([1, 5, 10]):
	# 	tau_sigma(M_raw, all_the_labels, number_of_windows, tau_window, tmin, 'output_figures/Fig' + str(iteration_id + i + 1))

	### Compute the trasition matrix
	# T_matrix = compute_transition_matrix(all_the_labels, 'output_figures/Fig8')
	# normalized_T_matrix = normalize_T_matrix(T_matrix)

	### Print the file to color the MD trajectory on ovito
	# print_mol_labels1(all_the_labels, tau_window, 'all_cluster_IDs.dat')

if __name__ == "__main__":
	main()
