def plot_partial_trajectories(M, M1, T, all_the_labels, offset, list_popt, list_th, tau_delay, tau_window, filename):
	flat_M = M1.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	number_of_windows = int(T/tau_window)
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))
	
	big_pile_of_everything = [ [] for _ in range(number_of_windows) ]
	big_pile_of_labels = [ [] for _ in range(number_of_windows) ]
	for i, l in enumerate(all_the_labels):
		if i%50 == 0:
			for w in range(len(l)):
				if l[w] > offset and l[w] <= offset + len(list_popt):
					x_w = M[i][w*tau_window:(w + 1)*tau_window]
					big_pile_of_everything[w].append(x_w)
					big_pile_of_labels[w].append(l[w])

	THE_BIG_TIME_ARRAY = []
	THE_BIG_SIGNAL_ARRAY = []
	THE_BIG_COLOR_ARRAY = []
	for w, t_slice in enumerate(big_pile_of_everything):
		X = np.array(t_slice)
		time = np.linspace((tau_delay + w*tau_window)*t_conv, (tau_delay + (w + 1)*tau_window)*t_conv, tau_window)
		time = np.tile(time, len(X))
		color = np.repeat(big_pile_of_labels[w], tau_window)
		signal = X.flatten()
		THE_BIG_TIME_ARRAY = np.concatenate((THE_BIG_TIME_ARRAY, time))
		THE_BIG_SIGNAL_ARRAY = np.concatenate((THE_BIG_SIGNAL_ARRAY, signal))
		THE_BIG_COLOR_ARRAY = np.concatenate((THE_BIG_COLOR_ARRAY, color))
	ax[0].scatter(THE_BIG_TIME_ARRAY, THE_BIG_SIGNAL_ARRAY, c=THE_BIG_COLOR_ARRAY, vmin=offset+1, vmax=offset+len(list_popt), s=0.1, rasterized=True)
	ax[0].set_xlabel(r'Time ' + t_units)
	ax[0].set_ylabel(r'$t$SOAP signal ' + y_units)

	ax[1].stairs(counts, bins, fill=True, orientation='horizontal')
	for n, th in enumerate(list_th):
		ax[1].hlines(th, xmin=0.0, xmax=np.amax(counts), linestyle='--', color='black')
		ax[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *list_popt[n]), np.linspace(bins[0], bins[-1], 1000))

	ax[1].get_xaxis().set_visible(False)
	ax[0].set_ylim(tSOAP_lim)
	plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def plot_trajectories(M, T, all_the_labels, list_of_states, tau_delay, tau_window, filename):
	print('* Plotting colored trajectories...')
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=n_bins, density=True)
	time = np.linspace(tau_delay*t_conv, (T + tau_delay)*t_conv, T)
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 4.8))
	for i in range(len(M)):
		if len(M) < 100 or i%10 == 0:
			c = np.repeat(all_the_labels[i].flatten(), tau_window)
			T_max = c.size
			ax[0].scatter(time[:T_max], M[i][:T_max], c=c, s=0.05, alpha=0.1, rasterized=True)
	ax[0].set_xlabel(r'Time ' + t_units)
	ax[0].set_ylabel(r'$t$SOAP signal ' + y_units)

	## OPTION 1
	# for state in np.flip(list_of_states, axis=0):
	# 	id_inf = 0
	# 	id_sup = len(bins) - 1
	# 	for j in range(len(bins)):
	# 		if bins[j] >= state[0]:
	# 			id_inf = j
	# 			break
	# 	for j in range(len(bins)):
	# 		if bins[j] >= state[1]:
	# 			id_sup = j
	# 			break
	# 	ax[1].stairs(counts[id_inf+1:id_sup], bins[id_inf:id_sup], fill=True, orientation='horizontal', alpha=0.5)

	## OPTION 2
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal')
	color = ['black', 'blue', 'red', 'yellow', 'green', 'purple', 'gray', 'cyan']
	for i, state in enumerate(np.flip(list_of_states, axis=0)):
		ax[1].hlines(state[0], xmin=0.0, xmax=np.amax(counts), linestyle='--', color=color[i])
		ax[1].hlines(state[1], xmin=0.0, xmax=np.amax(counts), linestyle='--', color=color[i])

	ax[1].get_xaxis().set_visible(False)
	plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)
