import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
from functions import *

### System specific parameters ###
t_units = r'[ns]'			# Units of measure of time
# y_units = r'[$t$SOAP]'		# Units of measure of the signal

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
	M_raw = remove_first_points(M_raw, PAR[1])
	M = Savgol_filter(M_raw, PAR[3], poly_order)
	SIG_MAX = np.max(M)
	SIG_MIN = np.min(M)
	M = (M - SIG_MIN)/(SIG_MAX - SIG_MIN)
	total_time = M.shape[1]
	print('* Using ' + str(int(total_time/PAR[0])) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[2]) + ' ns). ')
	all_the_labels = np.zeros((len(M), int(total_time/PAR[0])))
	list_of_states = []

	### Create file for output
	with open(output_file, 'w') as f:
		print('# ' + str(PAR[0]) + ', ' + str(PAR[1]) + ', ' + str(PAR[2]), file=f)
	if not os.path.exists('output_figures'):
		os.makedirs('output_figures')

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

	return relabel_states(all_the_labels, list_of_states)

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
		ax[0].set_ylabel('Normalized signal')
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
	tau_delay = PAR[1]
	t_conv = PAR[2]
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

		ax.scatter(flat_times, flat_signals, c=flat_colors, vmin=0, vmax=np.amax(States), s=0.05)
	
	ax.set_xlabel('Time ' + t_units)
	ax.set_ylabel('Normalized signal')
	ax.set_ylim(y_lim)
	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def state_statistics(M, PAR, all_the_labels, resolution, filename):
	print('* Computing some statistics on the states...')
	tau_window = PAR[0]
	t_conv = PAR[2]
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
	ax.set_ylabel(r'State mean amplitude')
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
			tmp_label1.append('State ' + str(n) + ': ' + "{:.2f}".format(starting_fraction*100) + '%')
			tmp_label2.append('State ' + str(n) + ': ' + "{:.2f}".format(ending_fraction*100) + '%')

	label = np.concatenate((tmp_label1, tmp_label2))
	palette = sns.color_palette('viridis', n_colors=n_states-2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')
	color = np.tile(palette, frame_list.size)
	# x_loc = np.concatenate((np.zeros(n_states), np.ones(n_states)))
	# y_loc = np.empty(2*n_states)
	# y_sum1 = 0
	# y_sum2 = 0
	# delta = 0.1
	# for n in range(n_states):
	# 	y_loc[n] = y_sum1
	# 	y_loc[n_states + n] = y_sum2
	# 	y_sum1 += (np.sum(T[n]) + delta)/(np.sum(T) + n_states*delta)
	# 	y_sum2 += (np.sum(T.T[n]) + delta)/(np.sum(T) + n_states*delta)
	# node = dict(label=label, x=x_loc, y=y_loc, pad=30, thickness=20, color=color)
	node = dict(label=label, pad=30, thickness=20, color=color)
	link = dict(source=source, target=target, value=value)
	Data = go.Sankey(link=link, node=node, arrangement="perpendicular")
	fig = go.Figure(Data)
	# fig.update_layout(title='Tau = ' + str(t_jump*t_conv) + ' ns')

	if show_plot:
		fig.show()
	fig.write_image(filename + '.png', scale=5.0)

def compute_transition_matrix(PAR, all_the_labels, filename):
	tau_window = PAR[0]
	t_conv = PAR[2]
	unique_labels = np.unique(all_the_labels)
	n_states = unique_labels.size
	
	T = np.zeros((n_states, n_states))
	for L in all_the_labels:
		for w in range(len(L) - 1):
			ID0 = int(L[w])
			ID1 = int(L[w + 1])
			T[ID0][ID1] += 1

	T_sym = np.divide(T + np.transpose(T), 2.0)
	T = normalize(T_sym, axis=1, norm='l1')

	fig, ax = plt.subplots(figsize=(10, 8))
	T_plot = copy.deepcopy(T)
	T_min = T[0][0]
	for a in range(T_plot.shape[0]):
		for b in range(T_plot.shape[1]):
			if T_plot[a][b] < T_min and T_plot[a][b] > 0:
				T_min = T_plot[a][b]
	for a in range(T_plot.shape[0]):
		for b in range(T_plot.shape[1]):
			if T_plot[a][b] == 0.0:
				T_plot[a][b] = T_min
	im = ax.imshow(T_plot, norm=LogNorm(vmin=np.min(T_plot), vmax=np.max(T_plot)))
	fig.colorbar(im)
	for (i, j),val in np.ndenumerate(T_plot):
		ax.text(j, i, "{:.2f}".format(100*val), ha='center', va='center')
	fig.suptitle(r'$\tau=$' + str(tau_window*t_conv) + r' ns')
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

def main():
	M_raw, M, PAR, all_the_labels, list_of_states = all_the_input_stuff()

	all_the_labels, list_of_states = iterative_search(M, PAR, all_the_labels, list_of_states)

	# plot_all_trajectories(M, PAR, all_the_labels, list_of_states, 'output_figures/Fig2_')
	# y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	# plot_one_trajectory(M[PAR[5]], PAR, all_the_labels[PAR[5]], list_of_states, np.unique(all_the_labels), y_lim, 'output_figures/Fig3')

	# state_statistics(M, PAR, all_the_labels, 1, 'output_figures/Fig4')

	for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 100]), np.array([0, 50, 100])]):
		sankey(all_the_labels, frame_list, sankey_average, PAR[2], 'output_figures/Fig5_' + str(i))
	# compute_transition_matrix(PAR, all_the_labels, 'output_figures/Fig6')

	# print_mol_labels1(all_the_labels, PAR, 'all_cluster_IDs.dat')

if __name__ == "__main__":
	main()
