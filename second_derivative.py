import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
import pycwt as wavelet
from functions import *

### System specific parameters ###
t_units = r'[ns]'			# Units of measure of time

### Usually no need to changhe these ###
output_file = 'states_output.txt'
poly_order = 2 				# Savgol filter polynomial order
n_bins = 100 				# Number of bins in the histograms
stop_th = 0.001				# Treshold to exit the maxima search

sankey_average = 10			# On how many frames to average the Sankey diagrams
show_plot = True			# Show all the plots

def all_the_input_stuff():
	### Read and clean the data points
	data_directory, PAR = read_input_parameters()
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])
	M_raw = remove_edges(M_raw, PAR[1])
	M_raw = Savgol_filter(M_raw, PAR[3], poly_order)

	### Compute the derivative of the signal
	tmp_M = []
	for x in M_raw:
		tmp_x = np.diff(x)/PAR[2]
		tmp_M.append(tmp_x)
	M = np.array(tmp_M)

	# SIG_MAX = np.max(M)
	# SIG_MIN = np.min(M)
	# M = (M - SIG_MIN)/(SIG_MAX - SIG_MIN)
	total_time = M.shape[1]
	print('* Using ' + str(int(total_time/PAR[0])) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[2]) + ' ns). ')
	all_the_labels = np.zeros((len(M), int(total_time/PAR[0])))
	list_of_states = []

	fig, ax = plt.subplots()
	for i, x in enumerate(M):
		if i%10 == 0:
			ax.scatter(np.linspace(PAR[1]*PAR[2], (PAR[1] + x.size)*PAR[2], x.size), x, s=0.25, c='xkcd:black')
	ax.set_xlabel(r'Simulation time ' + t_units)
	ax.set_ylabel(r'Derivative of the signal')
	plt.show()

	### Create files for output
	with open(output_file, 'w') as f:
		print('# ' + str(PAR[0]) + ', ' + str(PAR[1]) + ', ' + str(PAR[2]), file=f)
	if not os.path.exists('output_figures'):
		os.makedirs('output_figures')

	return M_raw, M, PAR, all_the_labels, list_of_states

def gauss_fit_n(M, n_bins, number_of_sigmas, filename):
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)

	def moving_average(data, window):
	    weights = np.repeat(1.0, window) / window
	    return np.convolve(data, weights, mode='valid')

	counts = moving_average(counts, 3)
	bins = moving_average(bins, 3)

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
		if current_max == max_ID.size:
			break
	if len(tmp_min_ID) == max_ID.size:
		tmp_min_ID.append(min_ID[n + 1])

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

		### "Zoom in" into the relevant bins interval
		reflat = flat_M[(flat_M > bins[id0]) & (flat_M <= bins[id1])]
		recounts, rebins = np.histogram(reflat, bins=2*(id1-id0), density=True)

		### Perform the Gaussian fit
		p0 = [bins[max_ID[n]], (bins[min_ID[n + 1]] - bins[min_ID[n]])/6, counts[max_ID[n]]]
		try:
			popt, pcov = scipy.optimize.curve_fit(gaussian, rebins[:-1], recounts, p0=p0)
		except RuntimeError:
			print('\tgauss_fit_n: RuntimeError.')
			continue
		popt[2] *= reflat.size
		if popt[1] < 0:
			popt[1] = -popt[1]
		flag = 1
		if popt[0] < Bins[0] or popt[0] > Bins[-1]:
			flag = 0 # If mu is outside the fitting range, it's not identifying the right Gaussian. Discard. 
			print('\tgauss_fit_n: Unable to correctly fit a Gaussian.')
		if popt[1] > Bins[-1] - Bins[0]:
			flag = 0 # If sigma is larger than the fitting interval, it's not identifying the right Gaussian. Discard. 
			print('\tgauss_fit_n: Unable to correctly fit a Gaussian.')
		perr = np.sqrt(np.diag(pcov))
		for j in range(len(perr)):
			if perr[j]/popt[j] > 0.5:
				flag = 0 # If the uncertanties over the parameters is too large, discard.
				print('\tgauss_fit_n: Parameters uncertanty too large.')
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
	# for n in range(len(list_th) - 1):
	# 	if list_th[n][1] > list_th[n + 1][0]:
	# 		mu0 = list_popt[n][0]
	# 		sigma0 = list_popt[n][1]
	# 		mu1 = list_popt[n + 1][0]
	# 		sigma1 = list_popt[n + 1][1]
	# 		middle_th = (mu0/sigma0 + mu1/sigma1)/(1/sigma0 + 1/sigma1)
	# 		list_th[n][1] = middle_th
	# 		list_th[n + 1][0] = middle_th

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
		list_popt, list_th = gauss_fit_n(M1, n_bins, PAR[4], 'output_figures/Fig1_' + str(iteration_id))

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

	return relabel_states(all_the_labels, list_of_states, stop_th)

def plot_all_trajectories(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing colored trajectories with histograms...')
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	States = np.unique(all_the_labels)
	for c, S in enumerate(States):
		list_of_times = []
		list_of_signals = []
		list_of_times2 = []
		list_of_signals2 = []
		for i, L in enumerate(all_the_labels):
			if i%10!=0:
				continue
			for w, l in enumerate(L):
				t0 = w*tau_window
				t1 = (w + 1)*tau_window
				if l == S:
					list_of_times.append(np.linspace((tau_delay + t0)*t_conv, (tau_delay + t1)*t_conv, tau_window))
					list_of_signals.append(M[i][t0:t1])
				elif l > S:
					list_of_times2.append(np.linspace((tau_delay + t0)*t_conv, (tau_delay + t1)*t_conv, tau_window))
					list_of_signals2.append(M[i][t0:t1])

		list_of_times = np.array(list_of_times)
		list_of_signals = np.array(list_of_signals)
		if list_of_times.shape[0] > 10000:
			list_of_times = list_of_times[0::10]
			list_of_signals = list_of_signals[0::10]
		flat_times = list_of_times.flatten()
		flat_signals = list_of_signals.flatten()
		flat_colors = c*np.ones(flat_times.size)
		list_of_times2 = np.array(list_of_times2)
		list_of_signals2 = np.array(list_of_signals2)
		if list_of_times2.shape[0] > 10000:
			list_of_times2 = list_of_times2[0::10]
			list_of_signals2 = list_of_signals2[0::10]
		flat_times2 = list_of_times2.flatten()
		flat_signals2 = list_of_signals2.flatten()

		if c < States.size - 1:
			fig, ax = plt.subplots(2, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 8))
			fig.suptitle('State ' + str(c))
			t_lim = np.array([tau_delay, (tau_delay + M.shape[1])])*t_conv
			y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]

			ax[0][0].scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States), s=0.05, alpha=0.5, rasterized=True)
			ax[0][0].set_ylabel('Normalized signal')
			ax[0][0].set_xlim(t_lim)
			ax[0][0].set_ylim(y_lim)
			ax[0][1].stairs(counts, bins, fill=True, orientation='horizontal')
			if c < len(States) - 1:
				ax[0][1].hlines(list_of_states[c][1], xmin=0.0, xmax=np.amax(counts), linestyle='--', color='black')
				ax[0][1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *list_of_states[c][0]), np.linspace(bins[0], bins[-1], 1000))

			ax[1][0].scatter(flat_times2, flat_signals2, c='black', s=0.05, alpha=0.5, rasterized=True)
			ax[1][0].set_xlabel(r'Time ' + t_units)
			ax[1][0].set_ylabel('Normalized signal')
			ax[1][0].set_xlim(t_lim)
			ax[1][0].set_ylim(y_lim)
			counts2, bins2 = np.histogram(flat_signals2, bins=n_bins, density=True)
			counts2 *= flat_signals2.size
			ax[1][1].stairs(counts2, bins2, fill=True, orientation='horizontal')
		else:
			fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
			fig.suptitle('State ' + str(c))
			t_lim = np.array([tau_delay, (tau_delay + M.shape[1])])*t_conv
			y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]

			ax[0].scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States), s=0.05, alpha=0.5, rasterized=True)
			ax[0].set_ylabel('Normalized signal')
			ax[0].set_xlim(t_lim)
			ax[0].set_ylim(y_lim)
			ax[1].stairs(counts, bins, fill=True, orientation='horizontal')

		if show_plot:
			plt.show()
		fig.savefig(filename + str(c) + '.png', dpi=600)
		plt.close(fig)

def plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, filename):
	print('* Printing cumulative figure...')
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.8)

	palette = sns.color_palette('viridis', n_colors=np.unique(all_the_labels).size - 2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')

	States = np.unique(all_the_labels)
	t_lim = np.array([tau_delay, (tau_delay + M.shape[1])])*t_conv
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	
	for c, S in enumerate(States[:-1]):
		list_of_times = []
		list_of_signals = []
		for i, L in enumerate(all_the_labels):
			for w, l in enumerate(L):
				t0 = w*tau_window
				t1 = (w + 1)*tau_window
				if l == S:
					list_of_times.append(np.linspace((tau_delay + t0)*t_conv, (tau_delay + t1)*t_conv, tau_window))
					list_of_signals.append(M[i][t0:t1])

		list_of_times = np.array(list_of_times)
		list_of_signals = np.array(list_of_signals)
		if list_of_times.shape[0] > 10000:
			list_of_times = list_of_times[0::100]
			list_of_signals = list_of_signals[0::100]
		flat_times = list_of_times.flatten()
		flat_signals = list_of_signals.flatten()
		flat_colors = c*np.ones(flat_times.size)

		ax[0].scatter(flat_times, flat_signals, c='xkcd:black', vmin=0, vmax=np.amax(States), s=0.05, alpha=0.5, rasterized=True)
		ax[0].set_ylabel('Normalized signal')
		ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
		ax[0].set_xlim(t_lim)
		ax[0].set_ylim(y_lim)
		ax[1].set_xticklabels([])
		if c < len(States) - 1:
			ax[1].hlines(list_of_states[c][1], xmin=0.0, xmax=np.amax(counts), linestyle='--', color=palette[c])
			ax[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *list_of_states[c][0]), np.linspace(bins[0], bins[-1], 1000), color=palette[c])

	for c, S in enumerate(States[:-1]):
		times = np.linspace(t_lim[0], t_lim[1], 100)
		ax[0].fill_between(times, np.max([0.0, list_of_states[c][1][0]]), np.min([list_of_states[c][1][1], 1.0]), color=palette[c], alpha=0.25)

	plt.show()
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def plot_one_trajectory(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	x = M[PAR[5]]
	L = all_the_labels[PAR[5]]
	fig, ax = plt.subplots()
	for c, S in enumerate(np.unique(all_the_labels)):
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

		ax.scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(np.unique(all_the_labels)), s=1.0)
	
	fig.suptitle('Example particle: ID = ' + str(PAR[5]))
	ax.set_xlabel('Time ' + t_units)
	ax.set_ylabel('Normalized signal')
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	ax.set_ylim(y_lim)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def tau_sigma(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	resolution = PAR[6]
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
	axa.set_ylabel(r'Gaussian noise amplitude $\sigma_n$')
	figa.savefig(filename + 'a.png', dpi=600)

	figb, axb = plt.subplots(figsize=(7.5, 4.8))
	axb.scatter(tau_c, data[1], c=labels, s=1.0)
	axb.set_xlabel(r'Correlation time $\tau_c$ [ps]')
	axb.set_ylabel(r'Gaussian noise amplitude $\sigma_n$')
	# axb.legend()
	figb.savefig(filename + 'b.png', dpi=600)
	
	if show_plot:
		plt.show()

def state_statistics(M, PAR, all_the_labels, filename):
	print('* Computing some statistics on the enviroinments...')
	tau_window = PAR[0]
	t_conv = PAR[2]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	resolution = PAR[6]
	data = []
	data2 = []
	labels = []
	labels2 = []
	for i, x in enumerate(M):
		data_mol = []
		labels_mol = []
		current_label = all_the_labels[i][0]
		x_w = x[0:tau_window]
		for w in range(1, number_of_windows):
			if all_the_labels[i][w] == current_label:
				x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
			else:
				if x_w.size < tau_window*resolution:
			 		continue
				data.append([x_w.size*t_conv, np.mean(x_w), np.std(x_w)])
				labels.append(int(current_label))
				data_mol.append([x_w.size*t_conv, np.mean(x_w), np.std(x_w)])
				labels_mol.append(int(current_label))
				x_w = x[tau_window*w:tau_window*(w + 1)]
				current_label = all_the_labels[i][w]
		data2.append(np.array(data_mol))
		labels2.append(np.array(labels_mol))

	data = np.array(data)
	data_tr = data.T

	### Characterization of the states ###
	state_points = []
	for s in np.unique(labels):
		ID_s = np.where(labels == s)[0]
		T = np.mean(data_tr[0][ID_s])
		sigma_T = np.std(data_tr[0][ID_s])
		A = np.mean(data_tr[1][ID_s])
		sigma_A = np.std(data_tr[1][ID_s])
		S = np.mean(data_tr[2][ID_s])
		sigma_S = np.std(data_tr[2][ID_s])
		state_points.append([T, A, S, sigma_T, sigma_A, sigma_S])
	state_points_tr = np.array(state_points).T

	with open(output_file, 'a') as f:
		print('\nEnviroinments\n', file=f)
		for E in state_points:
			print(E, file=f)

	fig, ax = plt.subplots()
	scatter = ax.scatter(data_tr[0], data_tr[1], c=labels, s=1.0)
	ax.errorbar(state_points_tr[0], state_points_tr[1], xerr=state_points_tr[3], yerr=state_points_tr[4], marker='o', ms=3.0, c='red', lw=0.0, elinewidth=1.0, capsize=2.5)
	fig.suptitle('Dynamic enviroinment statistics')
	ax.set_xlabel(r'Duration $T$ [ns]')
	ax.set_ylabel(r'Amplitude $A$')
	ax.legend(*scatter.legend_elements())
	fig.savefig(filename + 'a.png', dpi=600)

	### Characterization of the transitions ###
	n_states = np.unique(all_the_labels).size

	### Create dictionary
	dictionary = np.empty((n_states, n_states))
	c = 0
	for i in range(n_states):
		for j in range(n_states):
			if i != j:
				dictionary[i][j] = c
				c += 1
	### Create reference legend table
	ref_legend_table = []
	for i in range(n_states):
		for j in range(n_states):
			if i != j:
				ref_legend_table.append(str(i) + '-->' + str(j))

	transition_data = []
	transition_labels = []
	for i in range(len(data2)):
		for t in range(data2[i].shape[0] - 1):
			transition_data.append([data2[i][t][0], data2[i][t + 1][1] - data2[i][t][1]])
			transition_labels.append(dictionary[labels2[i][t]][labels2[i][t + 1]])

	transition_data_tr = np.array(transition_data).T
	transition_labels = np.array(transition_labels)

	state_points = []
	for i in range(dictionary.shape[0]):
		for j in range(dictionary.shape[1]):
			s = dictionary[i][j]
			# for s in np.unique(dictionary):
			# for s in np.unique(transition_labels):
			ID_s = np.where(transition_labels == s)[0]
			if ID_s.size == 0:
				continue
			T = np.mean(transition_data_tr[0][ID_s])
			sigma_T = np.std(transition_data_tr[0][ID_s])
			A = np.mean(transition_data_tr[1][ID_s])
			sigma_A = np.std(transition_data_tr[1][ID_s])
			color = dictionary[i][j]
			if i > j:
				color = dictionary[j][i]
			state_points.append([T, A, sigma_T, sigma_A, color])
	state_points_tr = np.array(state_points).T

	with open(output_file, 'a') as f:
		print('\nTransitions\n', file=f)
		for E in state_points:
			print(E, file=f)

	###################################################################################
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	for tr in [1.0, 5.0, 8.0, 9.0]:
		tmp_data = transition_data_tr[0][transition_labels == tr]

		counts, bins, _ = ax1.hist(tmp_data, bins=int(np.sqrt(tmp_data.size)/2 + 1), density=True, histtype='step', label=ref_legend_table[int(tr)])
		try:
			pos_counts = []
			pos_bins = []
			i = 0
			while counts[i] > 0:
				pos_counts.append(counts[i])
				pos_bins.append(bins[i])
				i += 1
			pos_bins = np.array(pos_bins)
			pos_counts = np.array(pos_counts)
			start_from = int(pos_bins.size/3)

			log_counts = np.log(pos_counts)

			popt, pcov = np.polyfit(pos_bins[start_from:], log_counts[start_from:], 1, cov=True)
			print('Tau:', -1/popt[0], np.sqrt(pcov[0][0])/popt[0]**2, '\toppure', np.exp(-popt[1]))
			y = np.exp(popt[0]*pos_bins[start_from:] + popt[1])
			ax1.plot(pos_bins[start_from:], y, linestyle='--', c='xkcd:black')

			# popt, pcov = scipy.optimize.curve_fit(exponential, pos_bins[start_from:-1], pos_counts[start_from:-1])
			# print('Tau:', popt[0], np.sqrt(pcov[0][0]))
			# ax1.plot(pos_bins[start_from:], exponential(pos_bins[start_from:], *popt), linestyle='--', c='xkcd:black')
		except:
			print('FAILURE')

		counts, bins, _ = ax2.hist(tmp_data, bins='auto', density=True, histtype='step', cumulative=True, label=ref_legend_table[int(tr)])
		upper_bins = [bins[b] for b in range(counts.size) if counts[b] > 1 - np.exp(-1)]
		upper_counts = [counts[b] for b in range(counts.size) if counts[b] > 1 - np.exp(-1)]
		try:
			popt, pcov = scipy.optimize.curve_fit(cumulative_exp, upper_bins, upper_counts)
			print(popt[0], np.sqrt(pcov[0][0]))
			times = np.linspace(bins[0], bins[-1], 1000)
			ax2.plot(times, cumulative_exp(times, *popt), linestyle='--', c='xkcd:grey', lw=1.0)
		except:
			print('FAILURE')

	ax1.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	ax1.set_ylabel(r'Probability density function PDF$(\Delta t)$')
	ax1.set_yscale('log')
	ax1.legend(loc='upper right')
	fig1.savefig(filename + 'b.png', dpi=600)

	ax2.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	ax2.set_ylabel(r'Cumulative distribution function CDF$(\Delta t)$')
	ax2.set_xscale('log')
	ax2.hlines(1 - np.exp(-1), bins[0], 80, linestyle='--', color='xkcd:black')
	ax2.legend(loc='lower right')
	fig2.savefig(filename + 'c.png', dpi=600)
	###################################################################################

	fig, ax = plt.subplots()
	scatter = ax.scatter(transition_data_tr[0], transition_data_tr[1], c=transition_labels, s=1.0, cmap='plasma', alpha=0.5)
	ax.errorbar(state_points_tr[0], state_points_tr[1], xerr=state_points_tr[2], yerr=state_points_tr[3], marker='o', ms=3.0, c='black', lw=0.0, elinewidth=1.0, capsize=2.5)
	ax.scatter(state_points_tr[0], state_points_tr[1], marker='s', s=30.0, c=state_points_tr[4], cmap='tab10')
	fig.suptitle('Transitions statistics')
	ax.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	ax.set_ylabel(r'Transition amplitude $\Delta A$')
	handles, _ = scatter.legend_elements()
	tmp = []
	for fl in np.unique(transition_labels):
		tmp.append(ref_legend_table[int(fl)])
	tmp = np.array(tmp)
	ax.legend(handles, tmp)
	ax.set_xscale('log')
	fig.savefig(filename + 'd.png', dpi=600)

	plt.show()
	if show_plot:
		plt.show()

def sankey(all_the_labels, frame_list, aver_window, t_conv, filename):
	print('* Computing and plotting the averaged Sankey diagrams...')
	if frame_list[-1] + aver_window > all_the_labels.shape[1]:
		print('ERROR: the required frame range is out of bound.')
		return

	n_states = np.unique(all_the_labels).size
	source = np.empty((frame_list.size - 1)*n_states**2)
	target = np.empty((frame_list.size - 1)*n_states**2)
	value = np.empty((frame_list.size - 1)*n_states**2)
	c = 0
	tmp_label1 = []
	tmp_label2 = []

	for i, t0 in enumerate(frame_list[:-1]):
		t_jump = frame_list[i + 1] - frame_list[i]
		T = np.zeros((n_states, n_states))
		for t in range(t0, t0 + aver_window):
			for L in all_the_labels:
				T[int(L[t])][int(L[t + t_jump])] += 1

		for n1 in range(len(T)):
			for n2 in range(len(T[n1])):
				source[c] = n1 + i*n_states
				target[c] = n2 + (i + 1)*n_states
				value[c] = T[n1][n2]
				c += 1

		for n in range(n_states):
			starting_fraction = np.sum(T[n])/np.sum(T)
			ending_fraction = np.sum(T.T[n])/np.sum(T)
			if i == 0:
				tmp_label1.append('State ' + str(n) + ': ' + "{:.2f}".format(starting_fraction*100) + '%')
			tmp_label2.append('State ' + str(n) + ': ' + "{:.2f}".format(ending_fraction*100) + '%')

	label = np.concatenate((tmp_label1, np.array(tmp_label2).flatten()))
	palette = sns.color_palette('viridis', n_colors=n_states-2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')
	color = np.tile(palette, frame_list.size)

	node = dict(label=label, pad=30, thickness=20, color=color)
	link = dict(source=source, target=target, value=value)
	Data = go.Sankey(link=link, node=node, arrangement="perpendicular")
	fig = go.Figure(Data)
	fig.update_layout(title='Frames: ' + str(frame_list*t_conv) + ' ns')

	if show_plot:
		fig.show()
	fig.write_image(filename + '.png', scale=5.0)

def main():
	M_raw, M, PAR, all_the_labels, list_of_states = all_the_input_stuff()

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)

	plot_all_trajectories(M, PAR, all_the_labels, list_of_states, 'output_figures/Fig2_')
	plot_one_trajectory(M, PAR, all_the_labels, 'output_figures/Fig3')
	plot_cumulative_figure(M, PAR, all_the_labels, list_of_states, 'output_figures/Fig3_cumulative')

	state_statistics(M, PAR, all_the_labels, 'output_figures/Fig4')
	tau_sigma(M_raw, PAR, all_the_labels, 'output_figures/Fig5')

	for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 15, 30, 45, 60])]):
		sankey(all_the_labels, frame_list, sankey_average, PAR[2], 'output_figures/Fig6_' + str(i))

	print_mol_labels1(all_the_labels, PAR, 'all_cluster_IDs.dat')

if __name__ == "__main__":
	main()
