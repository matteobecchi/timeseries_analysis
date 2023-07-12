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
show_plot = False			# Show all the plots

def all_the_input_stuff():
	data_directory, PAR = read_input_parameters()

	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])

	M_raw = M_raw[:, PAR[1]:]
	M = Savgol_filter(M_raw, PAR[0], poly_order)
	SIG_MAX = np.max(M)
	SIG_MIN = np.min(M)
	M = (M - SIG_MIN)/(SIG_MAX - SIG_MIN)
	total_time = M.shape[1]
	print('* Using ' + str(int(total_time/PAR[0])) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[2]) + ' ns). ')
	all_the_labels = np.zeros((len(M), int(total_time/PAR[0])))
	list_of_states = []

	### Create files for output
	with open(output_file, 'w') as f:
		print('# ' + str(PAR[0]) + ', ' + str(PAR[1]) + ', ' + str(PAR[2]), file=f)
	if not os.path.exists('output_figures'):
		os.makedirs('output_figures')

	return M_raw, M, PAR, all_the_labels, list_of_states

def plot_input_data(M, PAR, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]

	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

	time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1], M.shape[1])*t_conv
	if M.shape[1] > 1000:
		for mol in M[::10]:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	ax[0].set_ylabel('Normalized signal')
	ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
	ax[1].set_xticklabels([])

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

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

def gauss_fit_sum(M, n_bins, filename):
	print('* Gaussian fit...')
	number_of_sigmas = 2.0
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=False)

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
	for n in range(1, len(min_ID)):
		if min_ID[n] > max_ID[current_max]:
			tmp_min_ID.append(min_ID[n])
			current_max += 1
		if current_max == max_ID.size:
			break
	if len(tmp_min_ID) == max_ID.size:
		tmp_min_ID.append(min_ID[n + 1])

	min_ID = np.array(tmp_min_ID)

	p0 = []
	for n in range(max_ID.size):
		p0.append(bins[max_ID[n]])
		p0.append((bins[min_ID[n + 1]] - bins[min_ID[n]])/6)
		p0.append(counts[max_ID[n]]*np.sqrt(np.pi)*(bins[min_ID[n + 1]] - bins[min_ID[n]])/6)

	bounds = [np.array([-np.inf, 0.0, 0.0]), np.inf]
	for n in range(max_ID.size - 1):
		bounds[0] = np.concatenate((bounds[0], np.array([-np.inf, 0.0, 0.0])))

	try:
		Popt, pcov = scipy.optimize.curve_fit(sum_of_Gaussians, bins[:-1], counts, p0=p0, bounds=bounds)
	except RuntimeError:
		print('\tgauss_fit_n: RuntimeError.')
		return [], []

	list_popt = []
	for i in range(0, Popt.size, 3):
		list_popt.append([Popt[i], Popt[i + 1], Popt[i + 2]])

	print('\tGaussians parameters:')
	with open(output_file, 'a') as f:
		print('\n', file=f)
		for popt in list_popt:
			print(f'\tmu = {popt[0]:.4f}, sigma = {popt[1]:.4f}, peak = {popt[2]/popt[1]/np.sqrt(np.pi):.4f}')
			print(f'\tmu = {popt[0]:.4f}, sigma = {popt[1]:.4f}, peak = {popt[2]/popt[1]/np.sqrt(np.pi):.4f}', file=f)

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
	x = np.linspace(bins[0], bins[-1], 1000)
	ax.plot(x, sum_of_Gaussians(x, *Popt))

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return list_popt, list_th

def gauss_fit_max(M, n_bins, filename):
	print('* Gaussian fit...')
	number_of_sigmas = 2.0
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
	for n in range(1, len(min_ID)):
		if min_ID[n] > max_ID[current_max]:
			tmp_min_ID.append(min_ID[n])
			current_max += 1
		if current_max == max_ID.size:
			break
	if len(tmp_min_ID) == max_ID.size:
		tmp_min_ID.append(min_ID[n + 1])

	min_ID = np.array(tmp_min_ID)

	list_popt = []
	fit_done = False
	while (fit_done == False and max_ID.size > 0):
		counts_max = np.max(counts[max_ID])
		n = np.where(counts[max_ID] == counts_max)[0][0]

		### Chose the intersection interval between
		### the width at half height and the minima surrounding the maximum
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
		mu0 = bins[max_ID[n]]
		sigma0 = (bins[min_ID[n + 1]] - bins[min_ID[n]])/6
		A0 = counts[max_ID[n]]*np.sqrt(np.pi)*sigma0
		try:
			popt, pcov = scipy.optimize.curve_fit(Gaussian, rebins[:-1], recounts, p0=[mu0, sigma0, A0])
		except RuntimeError:
			print('\tgauss_fit_n: RuntimeError.')
			max_ID = np.delete(max_ID, n)
			min_ID = np.delete(min_ID, n + 1)
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
			fit_done = True
			list_popt.append(popt)
		else:
			max_ID = np.delete(max_ID, n)
			min_ID = np.delete(min_ID, n + 1)

	print('\tGaussians parameters:')
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
		ax.plot(np.linspace(bins[0], bins[-1], 1000), Gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

	return list_popt, list_th

def gauss_fit_n(M, n_bins, filename):
	number_of_sigmas = 2.0
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
			popt, pcov = scipy.optimize.curve_fit(Gaussian, rebins[:-1], recounts, p0=p0)
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
		ax.plot(np.linspace(bins[0], bins[-1], 1000), Gaussian(np.linspace(bins[0], bins[-1], 1000), *tmp_popt))
	# for th in list_th:
	# 	ax.vlines(th, 0, np.max(counts), linestyle='--', color='black')
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
			print(f'\tFraction of windows in state ' + str(offset + n) + f' = {fw:.3}')
			print(f'\tFraction of windows in state ' + str(offset + n) + f' = {fw:.3}', file=f)
			list_of_states[len(list_of_states) - len(counter) + n][2] = fw
	return np.array(M2), np.sum(counter)/(len(M)*number_of_windows), list_of_states

def iterative_search(M, PAR, all_the_labels, list_of_states):
	M1 = M
	iteration_id = 1
	states_counter = 0
	while True:
		### Locate and fit maxima in the signal distribution
		# list_popt, list_th = gauss_fit_sum(M1, n_bins, 'output_figures/Fig1_' + str(iteration_id))
		list_popt, list_th = gauss_fit_max(M1, n_bins, 'output_figures/Fig1_' + str(iteration_id))
		# list_popt, list_th = gauss_fit_n(M1, n_bins, 'output_figures/Fig1_' + str(iteration_id))

		for n in range(len(list_th)):
			list_of_states.append([list_popt[n], list_th[n], 0.0])

		### Find the windows in which the trajectories are stable in one maxima
		M2, c, list_of_states = find_stable_trj(M, PAR[0], list_th, list_of_states, all_the_labels, states_counter)

		states_counter += len(list_popt)
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c <= 0.0:
			break
		else:
			M1 = M2

	return relabel_states(all_the_labels, list_of_states)

def plot_cumulative_figure(M, PAR, list_of_states, final_list, filename):
	print('* Printing cumulative figure...')
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	n_states = len(list_of_states)
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	counts *= flat_M.size

	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

	palette = sns.color_palette('viridis', n_colors=n_states - 2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')

	t_lim = np.array([tau_delay + int(tau_window/2), (tau_delay + int(tau_window/2) + M.shape[1])])*t_conv
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	time = np.linspace(t_lim[0], t_lim[1], M.shape[1])

	if M.shape[1] > 1000:
		for mol in M[::10]:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	for S in range(n_states):
		ax[1].plot(Gaussian(np.linspace(bins[0], bins[-1], 1000), *list_of_states[S][0]), np.linspace(bins[0], bins[-1], 1000), color=palette[S])

	for n, th in enumerate(final_list):
		if th[1] == 0:
			ax[1].hlines(th[0], xmin=0.0, xmax=np.amax(counts), color='xkcd:black')
		elif th[1] == 1:
			ax[1].hlines(th[0], xmin=0.0, xmax=np.amax(counts), linestyle='--', color='xkcd:black')
		elif th[1] == 2:
			ax[1].hlines(th[0], xmin=0.0, xmax=np.amax(counts), linestyle='--', color='xkcd:red')
		if n < len(final_list) - 1:
			times = np.linspace(t_lim[0], t_lim[1], 100)
			ax[0].fill_between(times, final_list[n][0], final_list[n + 1][0], color=palette[n], alpha=0.25)

	ax[0].set_ylabel('Normalized signal')
	ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
	ax[0].set_ylim(y_lim)
	ax[1].set_xticklabels([])

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def plot_one_trajectory(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	example_ID = PAR[3]

	fig, ax = plt.subplots()
	t_lim = np.array([tau_delay + int(tau_window/2), (tau_delay + int(tau_window/2) + M.shape[1])])*t_conv
	times = np.linspace(t_lim[0], t_lim[1], M.shape[1])
	signal = M[example_ID]
	color = all_the_labels[example_ID]
	ax.scatter(times, signal, c=color, vmin=0, vmax=np.max(np.unique(all_the_labels)), s=1.0)

	fig.suptitle('Example particle: ID = ' + str(example_ID))
	ax.set_xlabel('Time ' + t_units)
	ax.set_ylabel('Normalized signal')
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def plot_all_trajectory_with_histos(M, PAR, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]

	fig = plt.figure()
	ax0 = plt.subplot(2, 4, 1)
	ax1 = plt.subplot(2, 4, 2)
	ax2 = plt.subplot(2, 4, 3)
	ax3 = plt.subplot(2, 4, 4)
	ax4 = plt.subplot(2, 1, 2)
	axes = [ax0, ax1, ax2, ax3, ax4]

	t_lim = np.array([tau_delay + int(tau_window/2), (tau_delay + int(tau_window/2) + M.shape[1])])*t_conv
	time = np.linspace(t_lim[0], t_lim[1], M.shape[1])

	if M.shape[1] > 1000:
		for mol in M[::10]:
			ax4.plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax4.plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	block_t = int(M.shape[1]/4)
	for i in range(4):
		part_signal = M[:, :(i + 1)*block_t].flatten()
		counts, bins = np.histogram(part_signal, bins='auto', density=True)
		axes[i].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)
		if i > 0:
			axes[i].set_yticklabels([])

	fig.suptitle('Example particle: ID = ' + str(PAR[3]))
	ax4.set_xlabel('Time ' + t_units)
	ax4.set_ylabel('Normalized signal')
	ax4.set_xlim(t_lim)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

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

def transition_matrix(Delta, t_conv, all_the_labels, filename):
	print('* Computing transition matrix...')
	n_states = np.unique(all_the_labels).size
	T = np.zeros((n_states, n_states))

	for mol in all_the_labels:
		for t in range(mol.size - Delta):
			id0 = int(mol[t])
			id1 = int(mol[t + Delta])
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
	fig.suptitle(r'Transition probabilities, $\Delta t=$' + str(Delta*t_conv) + ' ' + t_units)
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
				### fitting with x_n = \alpha*x_{n-1} + z_n, z_n Gaussian white noise
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
	data = []
	data2 = []
	labels = []
	labels2 = []
	for i, x in enumerate(M):
		data_mol = []
		labels_mol = []
		current_label = all_the_labels[i][0]
		x_t = np.array([M[i][0]])
		for t in range(1, M.shape[1]):
			if all_the_labels[i][t] == current_label:
				x_t = np.append(x_t, M[i][t])
			else:
				data.append([x_t.size*t_conv, np.mean(x_t), np.std(x_t)])
				labels.append(int(current_label))
				data_mol.append([x_t.size*t_conv, np.mean(x_t), np.std(x_t)])
				labels_mol.append(int(current_label))
				x_t = np.array([M[i][t]])
				current_label = all_the_labels[i][t]
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

	plt.show()
	if show_plot:
		plt.show()

def transition_statistics(M, PAR, all_the_labels, list_of_states, filename):
	print('* Computing some statistics on the transitions...')
	tau_window = PAR[0]
	t_conv = PAR[2]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	resolution = PAR[6]
	data = []
	labels = []
	for i, x in enumerate(M):
		data_mol = []
		labels_mol = []
		current_label = all_the_labels[i][0]
		x_w = x[0:tau_window]
		for w in range(1, number_of_windows):
			if all_the_labels[i][w] == current_label:
				x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
			else:
				data_mol.append(x_w)
				labels_mol.append(int(current_label))
				x_w = x[tau_window*w:tau_window*(w + 1)]
				current_label = all_the_labels[i][w]
		data.append(data_mol)
		labels.append(np.array(labels_mol))

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
	for i in range(len(data)):
		for t in range(len(data[i]) - 1):
			state0 = labels[i][t]
			state1 = labels[i][t + 1]
			if state0 == len(list_of_states) or state1 == len(list_of_states):
				continue
			mu0 = list_of_states[state0][0][0]
			mu1 = list_of_states[state1][0][0]
			sigma0 = list_of_states[state0][0][1]
			sigma1 = list_of_states[state1][0][1]
			min_dist0 = np.abs(np.mean(data[i][t]) - mu0)
			min_dist1 = np.abs(np.mean(data[i][t + 1]) - mu1)
			if (min_dist0 < sigma0 and min_dist1 < sigma1):
				transition_data.append([data[i][t].size*t_conv, np.mean(data[i][t + 1]) - np.mean(data[i][t])])
				transition_labels.append(dictionary[state0][state1])

	transition_data_tr = np.array(transition_data).T
	transition_labels = np.array(transition_labels)

	state_points = []
	for i in range(dictionary.shape[0]):
		for j in range(dictionary.shape[1]):
			s = dictionary[i][j]
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
	for tr in [1.0, 4.0, 6.0, 7.0]:
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
			print('Tau:', -1/popt[0], np.sqrt(pcov[0][0])/popt[0]**2)
			y = np.exp(popt[0]*pos_bins[start_from:] + popt[1])
			ax1.plot(pos_bins[start_from:], y, linestyle='--', c='xkcd:grey')
		except:
			print('FAILURE')

		counts, bins, _ = ax2.hist(tmp_data, bins='auto', density=True, histtype='step', cumulative=True, label=ref_legend_table[int(tr)])
		upper_bins = [bins[b] for b in range(counts.size)]# if counts[b] > 1 - np.exp(-1)]
		upper_counts = [counts[b] for b in range(counts.size)]# if counts[b] > 1 - np.exp(-1)]
		try:
			popt, pcov = scipy.optimize.curve_fit(cumulative_exp, upper_bins, upper_counts, p0=[1.0, 1.0])
			print(popt)
			print(np.sqrt(pcov))
			times = np.linspace(bins[0], bins[-1], 1000)
			ax2.plot(times, cumulative_exp(times, *popt2), linestyle='--', c='xkcd:gray', lw=1.0)
		except:
			print('FAILURE')

	ax1.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	ax1.set_ylabel(r'Probability density function PDF$(\Delta t)$')
	ax1.set_yscale('log')
	ax1.legend(loc='upper right')
	fig1.savefig(filename + 'a.png', dpi=600)

	ax2.set_xlabel(r'Waiting time $\Delta t$ [ns]')
	ax2.set_ylabel(r'Cumulative distribution function CDF$(\Delta t)$')
	ax2.set_xscale('log')
	ax2.legend(loc='lower right')
	fig2.savefig(filename + 'b.png', dpi=600)
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
	fig.savefig(filename + 'c.png', dpi=600)

	plt.show()
	if show_plot:
		plt.show()

def main():
	M_raw, M, PAR, all_the_labels, list_of_states = all_the_input_stuff()
	# plot_input_data(M, PAR, 'output_figures/Fig0')

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)
	list_of_states, final_list = set_final_states(list_of_states)
	all_the_labels = assign_final_states_to_single_frames(M, final_list)

	plot_cumulative_figure(M, PAR, list_of_states, final_list, 'output_figures/Fig2')
	# plot_all_trajectory_with_histos(M, PAR, 'output_figures/Fig2a')
	plot_one_trajectory(M, PAR, all_the_labels, 'output_figures/Fig3')

	print_mol_labels_fbf_gro(all_the_labels, PAR, 'all_cluster_IDs.dat')
	print_mol_labels_fbf_xyz(all_the_labels, PAR, 'all_cluster_IDs_xyz.dat')

	for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 100, 200, 300, 400])]):
		sankey(all_the_labels, frame_list, 10, PAR[2], 'output_figures/Fig4_' + str(i))

	transition_matrix(1, PAR[2], all_the_labels, 'output_figures/Fig5')
	state_statistics(M, PAR, all_the_labels, 'output_figures/Fig6')

if __name__ == "__main__":
	main()
