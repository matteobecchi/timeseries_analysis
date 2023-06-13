import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
import pycwt as wavelet
from functions import *

### Parameters to set
tau_smooth = 300			# Set the smoothing window # 300
tau_delay = 20000			# Remove the first tau_delay frames
tau_window = tau_smooth		# Size of the window for the "frame by frame" analysis.
number_of_sigmas = 1.0		# Set the treshold on the gaussian fit

### Other stuff, usually no need to changhe these ###
poly_order = 2 				# Savgol filter polynomial order
n_bins = 100 				# Number of bins in the histograms
stop_th = 0.01 				# Treshold to exit the maxima search
t_units = r'[ns]'			# Units of measure of time
t_conv = 0.001 				# Conversion between frames and time units
y_units = r'[$t$SOAP]'		# Units of measure of the signal
replot = True				# Plot all the data distribution during the maxima search

def all_the_input_stuff():
	### Read and clean the data points
	data_directory = read_input_parameters()
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])
	M_raw = remove_first_points(M_raw, tau_delay)
	M = Savgol_filter(M_raw, tau_smooth, poly_order)
	return M_raw, M

def gauss_fit_n(M, n_bins, filename):
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
			continue
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
	for popt in list_popt:
		print(f'\tmu = {popt[0]:.4f}, sigma = {popt[1]:.4f}, amplitude = {popt[2]:.4f}')

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
		for popt in list_popt:
			ax.plot(np.linspace(bins[0], bins[-1], 1000), gaussian(np.linspace(bins[0], bins[-1], 1000), *popt))
		for th in list_th:
			ax.vlines(th, 0, 100, linestyle='--', color='black')
		plt.show()
		fig.savefig(filename + '.png', dpi=600)

	return list_popt, list_th

def find_stable_trj(M, list_th, number_of_windows, all_the_labels, offset):
	list_th[0][0] = -np.inf
	list_th[len(list_th) - 1][1] = np.inf
	M2 = []
	counter = [ 0 for n in range(len(list_th)) ]
	for i, x in enumerate(M):
		for w in range(number_of_windows):
			x_w = x[w*tau_window:(w + 1)*tau_window]
			flag = 1
			for l, th in enumerate(list_th):
				if np.amin(x_w) > th[0] and np.amax(x_w) < th[1]:
					if all_the_labels[i][w] < 0.5:
						all_the_labels[i][w] = l + offset + 1
						counter[l] += 1
						flag = 0
						break
			if flag:
				M2.append(x_w)

	print('* Finding stable windows...')
	for n, c in enumerate(counter):
		fw = c/(len(M)*number_of_windows)
		print(f'\tFraction of windows in state ' + str(offset + n + 1) + f' = {fw:.3}')
	return np.array(M2), np.sum(counter)/(len(M)*number_of_windows)

def plot_trajectories(M, T, all_the_labels, filename):
	print('* Plotting colored trajectories...')
	time = np.linspace(tau_delay*t_conv, (T + tau_delay)*t_conv, T)
	fig, ax = plt.subplots()
	# for i in IDs_to_plot:
	for i in range(len(M)):
		if i%10 == 0:
			c = np.repeat(all_the_labels[i].flatten(), tau_window)
			T_max = c.size
			ax.scatter(time[:T_max], M[i][:T_max], c=c, s=0.05, alpha=0.1, rasterized=True)
	ax.set_xlabel(r'Time ' + t_units)
	ax.set_ylabel(r'$t$SOAP signal ' + y_units)
	plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

### TO DELETE ###
# def amplitude_vs_time(M, all_the_labels, number_of_windows, filename):
# 	data = []
# 	labels = []
# 	for i, x in enumerate(M):
# 		current_label = all_the_labels[i][0]
# 		x_w = x[0:tau_window]
# 		for w in range(1, number_of_windows):
# 			 if all_the_labels[i][w] == current_label:
# 			 	x_w = np.concatenate((x_w, x[tau_window*w:tau_window*(w + 1)]))
# 			 else:
# 			 	data.append([x_w.size, np.mean(x_w), np.std(x_w)])
# 			 	labels.append(current_label)
# 			 	x_w = x[tau_window*w:tau_window*(w + 1)]
# 			 	current_label = all_the_labels[i][w]

# 	data = np.array(data).T

# 	fig1, ax1 = plt.subplots()
# 	ax1.scatter(data[0]*t_conv, data[1], c=labels, s=1.0)
# 	ax1.set_xlabel(r'Duration $T$ ' + t_units)
# 	ax1.set_ylabel(r'Average amplitude ' + y_units)
# 	# ax1.legend()
# 	fig1.savefig(filename + 'a.png', dpi=600)

# 	fig2, ax2 = plt.subplots()
# 	ax2.scatter(data[0]*t_conv, data[2], c=labels, s=1.0)
# 	ax2.set_xlabel(r'Duration $T$ ' + t_units)
# 	ax2.set_ylabel(r'Standard deviation ' + y_units)
# 	# ax2.legend()
# 	fig2.savefig(filename + 'b.png', dpi=600)

# 	fig3, ax3 = plt.subplots()
# 	ax3.scatter(data[1], data[2], c=labels, s=1.0)
# 	ax3.set_xlabel(r'Average amplitude ' + y_units)
# 	ax3.set_ylabel(r'Standard deviation ' + y_units)
# 	# ax3.legend()
# 	fig3.savefig(filename + 'c.png', dpi=600)
# 	plt.show()

def tau_sigma(M, all_the_labels, number_of_windows, filename):
	print('* Computing the amplitude - correlation diagram...')
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
			 	if x_w.size < tau_window*10:
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
	M_raw, M = all_the_input_stuff()
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
	print('* Using ' + str(number_of_windows) + ' windows of length ' + str(tau_window) + ' frames (' + str(tau_window*t_conv) + ' ns). ')
	all_the_labels = np.array([ np.zeros(number_of_windows) for _ in range(len(M)) ])

	M1 = M
	iteration_id = 1
	states_counter = 0
	while True:
		### Locate and fit maxima in the signal distribution
		list_popt, list_th = gauss_fit_n(M1, n_bins, 'output_figures/Fig' + str(iteration_id))

		### Find the windows in which the trajectories are stable in one maxima
		M2, c = find_stable_trj(M, list_th, number_of_windows, all_the_labels, states_counter)

		states_counter += len(list_popt)
		iteration_id += 1
		### Exit the loop if no new stable windows are found
		if c < stop_th:
			break
		else:
			M1 = M2

	### Plot the histogram of the singnal remaining after the "onion" analysis
	if replot:
		plot_and_save_histogram(M2, n_bins, 'output_figures/Fig' + str(iteration_id))

	### Plot an example trajectory with the different colors
	plot_trajectories(M, T, all_the_labels, 'output_figures/Fig' + str(iteration_id + 1))

	### Amplitude vs time of the windows scatter plot
	tau_sigma(M_raw, all_the_labels, number_of_windows, 'output_figures/Fig' + str(iteration_id + 2))

	### Print the file to color the MD trajectory on ovito
	print_mol_labels1(all_the_labels, tau_window, 'all_cluster_IDs.dat')

if __name__ == "__main__":
	main()
