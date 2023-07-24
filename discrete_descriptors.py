import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from functions import *

### System specific parameters ###
t_units = r'[ns]'	# Units of measure of time

### Usually no need to changhe these ###
output_file = 'states_output.txt'
show_plot = True

def all_the_input_stuff():
	# Read input parameters from files.
	data_directory, PAR = read_input_parameters()

	# Read raw data from the specified directory/files.
	if type(data_directory) == str:
		M = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])

	# Remove initial frames based on 'tau_delay'.
	M = M[:, PAR[1]:]

	# Normalize the data to the range [0, 1].
	SIG_MAX = np.max(M)
	SIG_MIN = np.min(M)
	M = (M - SIG_MIN)/(SIG_MAX - SIG_MIN)

	# Get the number of particles and total frames in the trajectory.
	total_particles = M.shape[0]
	total_time = M.shape[1]
	total_states = np.unique(M).size

	# Calculate the number of windows for the analysis.
	num_windows = int(total_time / PAR[0])

	# Print informative messages about trajectory details.
	print('\tTrajectory has ' + str(total_particles) + ' particles. ')
	print('\tTrajectory of length ' + str(total_time) + ' frames (' + str(total_time*PAR[2]) + ' ns). ')
	print('\t' + str(total_states) + ' possible environments are identified. ')
	print('\tUsing ' + str(num_windows) + ' windows of length ' + str(PAR[0]) + ' frames (' + str(PAR[0]*PAR[2]) + ' ns). ')

	### Create files for output
	with open(output_file, 'w') as f:
		print('# ' + str(PAR[0]) + ', ' + str(PAR[1]) + ', ' + str(PAR[2]), file=f)
	if not os.path.exists('output_figures'):
		os.makedirs('output_figures')

	# Return required data for further analysis.
	return M, PAR, data_directory

def plot_input_data(M, PAR, filename):
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]

	# Flatten the M matrix and compute histogram counts and bins
	flat_M = M.flatten()
	counts, bins = np.histogram(flat_M, bins=15, density=True)
	counts *= flat_M.size

	# Create a plot with two subplots (side-by-side)
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

	# Plot histogram in the second subplot (right side)
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

	# Compute the time array for the x-axis of the first subplot (left side)
	time = np.linspace(tau_delay + int(tau_window/2), tau_delay + int(tau_window/2) + M.shape[1], M.shape[1])*t_conv

	# Plot the individual trajectories in the first subplot (left side)
    # If there are more than 1000 frames, plot only every 10th frame for faster rendering
	if M.shape[0]*M.shape[1] > 10000:
		for mol in M[::100]:
			ax[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5, rasterized=True)

	# Set labels and titles for the plots
	ax[0].set_ylabel('Normalized signal')
	ax[0].set_xlabel(r'Simulation time $t$ ' + t_units)
	ax[1].set_xticklabels([])

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def state_identification(M, PAR):
	tau_window = PAR[0]
	candidate_states = np.unique(M)

	valid_states = []
	for S in candidate_states:
		state_flag = 0
		for i in range(M.shape[0]):
			for w in range(int(M.shape[1]/tau_window)):
				flag = 1
				if all(xt == S for xt in M[i][w*tau_window:(w + 1)*tau_window]):
					valid_states.append(S)
					state_flag = 1
					break
			if state_flag == 1:
				break

	if valid_states[-1] < 1.0:
		valid_states.append(1.0)

	print('\t' + str(len(valid_states)) + ' valid states identified. ')

	th = []
	th.append(0.0)
	for n in range(len(valid_states) - 1):
		th.append((valid_states[n] + valid_states[n + 1])/2)
	th.append(1.0)

	all_the_labels = np.zeros(M.shape)
	for i, x in enumerate(M):
		for t, xt in enumerate(x):
			for c in range(len(th) - 1):
				if xt >= th[c] and xt <= th[c + 1]:
					all_the_labels[i][t] = c

	return valid_states, th, all_the_labels

def plot_cumulative_figure(M, PAR, list_of_states, th, data_directory, filename):
	print('* Printing cumulative figure...')
	tau_window = PAR[0]
	tau_delay = PAR[1]
	t_conv = PAR[2]
	n_states = len(list_of_states)

	# Flatten the M matrix and compute histogram counts and bins
	flat_M = M.flatten()
	bins = len(list_of_states)
	counts, bins = np.histogram(flat_M, bins=bins, density=True)
	counts *= flat_M.size

	# Create a subplot
	fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

	# Plot histogram in the second subplot (right side)
	ax[1].stairs(counts, bins, fill=True, orientation='horizontal', alpha=0.5)

	# Create a color palette for plotting states
	palette = sns.color_palette('viridis', n_colors=n_states - 2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')

	# Define time and y-axis limits for the left subplot (ax[0])
	t_lim = np.array([tau_delay + int(tau_window/2), (tau_delay + int(tau_window/2) + M.shape[1])])*t_conv
	y_lim = [np.min(M) - 0.025*(np.max(M) - np.min(M)), np.max(M) + 0.025*(np.max(M) - np.min(M))]
	time = np.linspace(t_lim[0], t_lim[1], M.shape[1])

	# Plot the individual trajectories on the left subplot (ax[0])
	if M.shape[0]*M.shape[1] > 10000:
		for mol in M[::100]:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)
	else:
		for mol in M:
			ax[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1, alpha=0.5, rasterized=True)

	# Plot the horizontal lines and shaded regions to mark list_of_states thresholds
	times = np.linspace(t_lim[0], t_lim[1], 100)
	for n in range(len(th) -1):
		ax[0].fill_between(times, th[n], th[n + 1], color=palette[n], alpha=0.25)

	# Set plot titles and axis labels
	fig.suptitle(data_directory)
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

	# Get the signal of the example particle
	signal = M[example_ID]

	# Create time values for the x-axis
	times = np.arange(tau_delay, tau_delay + M.shape[1]) * t_conv

	# Create a figure and axes for the plot
	fig, ax = plt.subplots()

	# Create a colormap to map colors to the labels of the example particle
	color = all_the_labels[example_ID]
	ax.plot(times, signal, c='black', lw=0.1)

	# Plot the signal as a line and scatter plot with colors based on the labels
	ax.scatter(times, signal, c=color, s=4.0, vmin=0, vmax=int(np.max(all_the_labels)))

	# Add title and labels to the axes
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

	# Check if the required frame range is within the bounds of the input data.
	if frame_list[-1] + aver_window > all_the_labels.shape[1]:
		print('\tERROR: the required frame range is out of bound.')
		return

	# Determine the number of unique states in the data.
	n_states = np.unique(all_the_labels).size

	# Create arrays to store the source, target, and value data for the Sankey diagram.
	source = np.empty((frame_list.size - 1) * n_states**2)
	target = np.empty((frame_list.size - 1) * n_states**2)
	value = np.empty((frame_list.size - 1) * n_states**2)

	# Initialize a counter variable.
	c = 0

	# Create temporary lists to store node labels for the Sankey diagram.
	tmp_label1 = []
	tmp_label2 = []

	# Loop through the frame_list and calculate the transition matrix for each time window.
	for i, t0 in enumerate(frame_list[:-1]):
		# Calculate the time jump for the current time window.
		t_jump = frame_list[i + 1] - frame_list[i]

		# Initialize a matrix to store the transition counts between states.
		T = np.zeros((n_states, n_states))
	    
		# Iterate through the current time window and increment the transition counts in T.
		for t in range(t0, t0 + aver_window):
			for L in all_the_labels:
				T[int(L[t])][int(L[t + t_jump])] += 1

		# Store the source, target, and value for the Sankey diagram based on T.
		for n1 in range(len(T)):
			for n2 in range(len(T[n1])):
				source[c] = n1 + i * n_states
				target[c] = n2 + (i + 1) * n_states
				value[c] = T[n1][n2]
				c += 1

		# Calculate the starting and ending fractions for each state and store node labels.
		for n in range(n_states):
			starting_fraction = np.sum(T[n]) / np.sum(T)
			ending_fraction = np.sum(T.T[n]) / np.sum(T)
			if i == 0:
				tmp_label1.append('State ' + str(n) + ': ' + "{:.2f}".format(starting_fraction * 100) + '%')
			tmp_label2.append('State ' + str(n) + ': ' + "{:.2f}".format(ending_fraction * 100) + '%')

	# Concatenate the temporary labels to create the final node labels.
	label = np.concatenate((tmp_label1, np.array(tmp_label2).flatten()))

	# Generate a color palette for the Sankey diagram.
	palette = sns.color_palette('viridis', n_colors=n_states - 2).as_hex()
	palette.insert(0, '#440154')
	palette.append('#fde725')

	# Tile the color palette to match the number of frames.
	color = np.tile(palette, frame_list.size)

	# Create dictionaries to define the Sankey diagram nodes and links.
	node = dict(label=label, pad=30, thickness=20, color=color)
	link = dict(source=source, target=target, value=value)

	# Create the Sankey diagram using Plotly.
	Data = go.Sankey(link=link, node=node, arrangement="perpendicular")
	fig = go.Figure(Data)

	# Add the title with the time information.
	fig.update_layout(title='Frames: ' + str(frame_list * t_conv) + ' ns')

	if show_plot:
		fig.show()
	fig.write_image(filename + '.png', scale=5.0)

def transition_matrix(Delta, t_conv, all_the_labels, filename):
	print('* Computing transition matrix...')
	# Get the number of unique states in the all_the_labels array.
	n_states = np.unique(all_the_labels).size
	# Initialize an empty matrix 'T' to store the transition counts.
	T = np.zeros((n_states, n_states))

	# Loop over each molecule trajectory in all_the_labels.
	for mol in all_the_labels:
		# Loop through the time steps of the trajectory.
		for t in range(mol.size - Delta):
			# Get the current state 'id0' and the state after 'Delta' steps 'id1'.
			id0 = int(mol[t])
			id1 = int(mol[t + Delta])
			# Increment the corresponding transition count in the matrix 'T'.
			T[id0][id1] += 1.0

	# Normalize the transition matrix 'T' to obtain transition probabilities 'N'.
	N = T / T.sum(axis=1, keepdims=True)

	# Create a plot to visualize the transition probabilities.
	fig, ax = plt.subplots(figsize=(10, 8))
	# Display the transition probabilities as an image with a logarithmic color scale.
	im = ax.imshow(N, cmap='viridis', norm=LogNorm(vmin=N[N > 0].min(), vmax=N.max()))
	fig.colorbar(im, ax=ax, format='%.2f')

	# Add text labels for each transition probability on the plot.
	for (i, j), val in np.ndenumerate(N):
		ax.text(j, i, "{:.2f}".format(val), ha='center', va='center')

	# Set titles and axis labels for the plot.
	fig.suptitle(r'Transition probabilities, $\Delta t=$' + str(Delta*t_conv) + ' ' + t_units)
	ax.set_xlabel('To...')
	ax.set_ylabel('From...')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_xticks(np.arange(n_states))
	ax.set_yticks(np.arange(n_states))
	ax.set_xticklabels(range(n_states))
	ax.set_yticklabels(range(n_states))
	plt.tight_layout()

	if show_plot:
		plt.show()
	fig.savefig(filename + '.png', dpi=600)
	plt.close(fig)

def tau_sigma(M, PAR, all_the_labels, filename):
	tau_window = PAR[0]
	T = M.shape[1]
	number_of_windows = int(T/tau_window)
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
	labels = []

	# Iterate through the molecules in M
	for i, x in enumerate(M):
		current_label = all_the_labels[i][0]  # Get the initial label of the molecule
		x_t = np.array([M[i][0]])  # Initialize a temporary array to store data for the current environment

		# Iterate through the frames of the current molecule
		for t in range(1, M.shape[1]):
			# Check if the current frame has the same label as the previous one
			if all_the_labels[i][t] == current_label:
				x_t = np.append(x_t, M[i][t])  # Add data to the temporary array
			else:
				data.append([x_t.size*t_conv, np.mean(x_t), np.std(x_t)])  # Store statistics for the environment
				labels.append(int(current_label))  # Store the label for the environment
				x_t = np.array([M[i][t]])  # Start a new temporary array for the next environment
				current_label = all_the_labels[i][t]  # Update the current label

	data = np.array(data)

	# Characterization of the states using numpy functions
	state_points = []
	for s in np.unique(labels):
		ID_s = np.where(labels == s)[0]
		state_data = data[ID_s]
		state_points.append([np.mean(state_data[:, 0]), np.mean(state_data[:, 1]), np.mean(state_data[:, 2]),
			np.std(state_data[:, 0]), np.std(state_data[:, 1]), np.std(state_data[:, 2])])
	state_points_tr = np.array(state_points).T

	# Append environment statistics to the output file
	with open(output_file, 'a') as f:
		print('\nEnviroinments\n', file=f)
		for E in state_points:
			print(E, file=f)

	# Create a scatter plot of the data points
	fig, ax = plt.subplots()
	scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=1.0, cmap='viridis')

	# Add error bars representing standard deviations of T and A
	ax.errorbar(state_points_tr[0], state_points_tr[1], xerr=state_points_tr[3], yerr=state_points_tr[4],
		marker='o', ms=3.0, c='red', lw=0.0, elinewidth=1.0, capsize=2.5)

	# Set plot titles and labels
	fig.suptitle('Dynamic enviroinment statistics')
	ax.set_xlabel(r'Duration $T$ [ns]')
	ax.set_ylabel(r'Amplitude $A$')
	ax.legend(*scatter.legend_elements())
	fig.savefig(filename + '.png', dpi=600)

	if show_plot:
		plt.show()

def transition_statistics(M, PAR, all_the_labels, list_of_states, filename):
	print('* Computing some statistics on the transitions...')
	tau_window = PAR[0]
	t_conv = PAR[2]
	data = []
	labels = []
	for i, x in enumerate(M):
		data_mol = []
		labels_mol = []
		current_label = all_the_labels[i][0]
		x_t = [x[0]]
		for t in range(1, M.shape[1]):
			if all_the_labels[i][t] == current_label:
				x_t.append(x[t])
			else:
				data_mol.append(np.array(x_t))
				labels_mol.append(int(current_label))
				x_t = [x[t]]
				current_label = all_the_labels[i][t]
		data.append(data_mol)
		labels.append(np.array(labels_mol))

	n_states = np.unique(all_the_labels).size

	### Create dictionary
	dictionary_21 = np.empty((n_states, n_states))
	dictionary_12 = []
	c = 0
	for i in range(n_states):
		for j in range(n_states):
			if i != j:
				dictionary_21[i][j] = c
				c += 1
				dictionary_12.append(str(i) + '-->' + str(j))

	transition_data = []
	transition_labels = []
	for i in range(len(data)):
		for t in range(len(data[i]) - 1):
			state0 = labels[i][t]
			state1 = labels[i][t + 1]
			mu0 = list_of_states[state0][0][0]
			mu1 = list_of_states[state1][0][0]
			sigma0 = list_of_states[state0][0][1]
			sigma1 = list_of_states[state1][0][1]
			transition_data.append([data[i][t].size*t_conv, np.mean(data[i][t + 1]) - np.mean(data[i][t])])
			transition_labels.append(dictionary_21[state0][state1])

	transition_data_tr = np.array(transition_data).T
	transition_labels = np.array(transition_labels)

	state_points = []
	for i in range(n_states):
		for j in range(n_states):
			s = dictionary_21[i][j]
			ID_s = np.where(transition_labels == s)[0]
			if ID_s.size == 0:
				continue
			T = np.mean(transition_data_tr[0][ID_s])
			sigma_T = np.std(transition_data_tr[0][ID_s])
			A = np.mean(transition_data_tr[1][ID_s])
			sigma_A = np.std(transition_data_tr[1][ID_s])
			color = dictionary_21[i][j]
			if i > j:
				color = dictionary_21[j][i]
			state_points.append([T, A, sigma_T, sigma_A, color])
	state_points_tr = np.array(state_points).T

	with open(output_file, 'a') as f:
		print('\nTransitions\n', file=f)
		for E in state_points:
			print(E, file=f)

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
		tmp.append(dictionary_12[int(fl)])
	tmp = np.array(tmp)
	ax.legend(handles, tmp)
	ax.set_xscale('log')
	fig.savefig(filename + '.png', dpi=600)

	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	for tr in [0.0, 2.0, 3.0, 5.0]:
		tmp_data = transition_data_tr[0][transition_labels == tr]
		counts, bins, _ = ax1.hist(tmp_data, bins='doane', density=True, histtype='step', label=dictionary_12[int(tr)])
		counts, bins, _ = ax2.hist(tmp_data, bins='auto', density=True, histtype='step', cumulative=True, label=dictionary_12[int(tr)])

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

	if show_plot:
		plt.show()

def main():
	M, PAR, data_directory = all_the_input_stuff()
	plot_input_data(M, PAR, 'output_figures/Fig0')

	list_of_states, th, all_the_labels = state_identification(M, PAR)

	plot_cumulative_figure(M, PAR, list_of_states, th, data_directory, 'output_figures/Fig2')
	plot_one_trajectory(M, PAR, all_the_labels, 'output_figures/Fig3')

	# print_mol_labels_fbf_gro(all_the_labels, 'all_cluster_IDs_gro.dat')
	print_mol_labels_fbf_xyz(all_the_labels, 'all_cluster_IDs_xyz.dat')

	# for i, frame_list in enumerate([np.array([0, 1]), np.array([0, 100, 200, 300])]):
	# 	sankey(all_the_labels, frame_list, 10, PAR[2], 'output_figures/Fig4_' + str(i))

	# state_statistics(M, PAR, all_the_labels, 'output_figures/Fig5')

if __name__ == "__main__":
	main()