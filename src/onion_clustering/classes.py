"""
Contains the classes used for storing parameters and system states.
"""
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
from matplotlib.ticker import MaxNLocator

from onion_clustering.first_classes import UniData, MultiData
from onion_clustering.first_classes import Parameters
from onion_clustering.first_classes import StateUni, StateMulti
from onion_clustering.functions import gaussian

class ClusteringObject:
    """This class contains input, output and methods for plotting."""

    def __init__(self, par: Parameters, data: UniData):
        self.par: Parameters = par
        self.data: UniData = data
        self.states: List[StateUni] = []
        self.number_of_states: np.ndarray
        self.fraction_0: np.ndarray

    def plot_input_data(self, filename: str):
        """Plots input data for visualization."""

        # Flatten the m_clean matrix and compute histogram counts and bins
        m_clean = self.data.matrix
        flat_m = m_clean.flatten()
        bins = self.par.bins
        counts, bins = np.histogram(flat_m, bins=bins, density=True)
        counts *= flat_m.size

        # Create a plot with two subplots (side-by-side)
        fig, axes = plt.subplots(1, 2, sharey=True,
            gridspec_kw={'width_ratios': [3, 1]},figsize=(9, 4.8))

        # Plot histogram in the second subplot (right side)
        axes[1].stairs(counts, bins, fill=True, orientation='horizontal')

        # Plot the individual trajectories in the first subplot (left side)
        time = self.par.print_time(m_clean.shape[1])
        step = 10 if m_clean.size > 1000000 else 1
        for mol in m_clean[::step]:
            axes[0].plot(time, mol, c='xkcd:black', lw=0.1, alpha=0.5,
                rasterized=True)

        # Set labels and titles for the plots
        axes[0].set_ylabel('Signal')
        axes[0].set_xlabel(r'Simulation time $t$ ' + self.par.t_units)
        axes[1].set_xticklabels([])

        fig.savefig('output_figures/' + filename + '.png', dpi=600)
        plt.close(fig)

    def preparing_the_data(self):
        """Processes raw data for analysis."""

        tau_window, t_smooth = self.par.tau_w, self.par.t_smooth
        t_conv, t_units = self.par.t_conv, self.par.t_units

        # Apply filtering on the data
        self.data.smooth_mov_av(t_smooth)  # Smoothing using moving average

        # Normalize the data to the range [0, 1]. Usually not needed. ###
        # self.data.normalize()

        # Calculate the number of windows for the analysis.
        num_windows = int(self.data.num_of_steps / tau_window)

        # Print informative messages about trajectory details.
        print('\tTrajectory has ' +
            str(self.data.num_of_particles) + ' particles. ')
        print('\tTrajectory of length ' + str(self.data.num_of_steps) +
            ' frames (' + str(self.data.num_of_steps*t_conv), t_units + ')')
        print('\tUsing ' + str(num_windows) + ' windows of length ' +
            str(tau_window) + ' frames (' +
            str(tau_window*t_conv), t_units + ')')

    def plot_state_populations(self):
        """Plots the populations of states over time.

        Steps:
        - Computes the populations of each state at different time steps.
        - Creates a plot illustrating state populations against time.
        - Utilizes Matplotlib to generate the plot based on provided data.
        - Saves the resulting plot as an image file.
        """
        print('* Printing populations vs time...')
        num_part = self.data.labels.shape[0]

        unique_labels = np.unique(self.data.labels)
        # If there are no assigned window, we still need the "0" state
        # for consistency:
        if 0 not in unique_labels:
            unique_labels = np.insert(unique_labels, 0, 0)

        list_of_populations = []
        for label in unique_labels:
            population = np.sum(self.data.labels == label, axis=0)
            list_of_populations.append(population / num_part)

        # Generate the color palette.
        palette = []
        n_states = unique_labels.size
        cmap = plt.get_cmap('viridis', n_states)
        for i in range(cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        fig, axes = plt.subplots()
        t_steps = self.data.labels.shape[1]
        time = self.par.print_time(t_steps)
        for label, pop in enumerate(list_of_populations):
            # pop_full = np.repeat(pop, self.par.tau_w)
            axes.plot(time, pop, label='ENV' + str(label), color=palette[label])
        axes.set_xlabel(r'Time ' + self.par.t_units)
        axes.set_ylabel(r'Population')
        axes.legend()

        fig.savefig('output_figures/Fig5.png', dpi=600)

    def sankey(self, tmp_frame_list: list[int]):
        """
        Computes and plots a Sankey diagram at the desired frames.

        Args:
        - tmp_frame_list (list[int]): List of frame indices.

        Steps:
        - Computes transition matrices for each time window based on label data.
        - Constructs source, target, and value arrays for the Sankey diagram.
        - Generates node labels and color palette for the diagram visualization.
        - Creates Sankey diagram using Plotly library and custom node/link data.
        - Saves the generated Sankey diagram as an image file.
        """
        print('* Computing and plotting the Sankey diagram...')

        all_the_labels = self.data.labels
        frame_list = np.array(tmp_frame_list)
        unique_labels = np.unique(all_the_labels)
        # If there are no assigned window, we still need the "0" state
        # for consistency:
        if 0 not in unique_labels:
            unique_labels = np.insert(unique_labels, 0, 0)
        n_states = unique_labels.size

        # Create arrays to store the source, target, and value data
        # for the Sankey diagram.
        source = np.empty((frame_list.size - 1) * n_states**2)
        target = np.empty((frame_list.size - 1) * n_states**2)
        value = np.empty((frame_list.size - 1) * n_states**2)

        # Initialize a counter variable.
        count = 0

        # Create temporary lists to store node labels for the Sankey diagram.
        tmp_label1 = []
        tmp_label2 = []

        # Loop through the frame_list and calculate the transition matrix
        # for each time window.
        for i, t_0 in enumerate(frame_list[:-1]):
            # Calculate the time jump for the current time window.
            t_jump = frame_list[i + 1] - frame_list[i]

            # Initialize a matrix to store the transition counts between states.
            trans_mat = np.zeros((n_states, n_states))

            # Iterate through the current time window and increment
            # the transition counts in trans_mat
            for label in all_the_labels:
                trans_mat[label[t_0]][label[t_0 + t_jump]] += 1

            # Store the source, target, and value for the Sankey diagram
            # based on trans_mat
            for j, row in enumerate(trans_mat):
                for k, elem in enumerate(row):
                    source[count] = j + i * n_states
                    target[count] = k + (i + 1) * n_states
                    value[count] = elem
                    count += 1

            # Calculate the starting and ending fractions for each state
            # and store node labels
            for j in range(n_states):
                start_fr = np.sum(trans_mat[j]) / np.sum(trans_mat)
                end_fr = np.sum(trans_mat.T[j]) / np.sum(trans_mat)
                if i == 0:
                    tmp_label1.append('State ' + str(j) + ': ' +
                        "{:.2f}".format(start_fr * 100) + '%')
                tmp_label2.append('State ' + str(j) + ': ' +
                    "{:.2f}".format(end_fr * 100) + '%')

        # Concatenate the temporary labels to create the final node labels.
        label = np.concatenate((tmp_label1, np.array(tmp_label2).flatten()))

        # Generate a color palette for the Sankey diagram.
        palette = []
        cmap = plt.get_cmap('viridis', n_states)
        for i in range(cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Tile the color palette to match the number of frames.
        color = np.tile(palette, frame_list.size)

        # Create dictionaries to define the Sankey diagram nodes and links.
        node = dict(label=label, pad=30, thickness=20, color=color)
        link = dict(source=source, target=target, value=value)

        # Create the Sankey diagram using Plotly.
        sankey_data = go.Sankey(link=link, node=node, arrangement="perpendicular")
        fig = go.Figure(sankey_data)

        # Add the title with the time information.
        fig.update_layout(title='Frames: ' + str(frame_list * self.par.t_conv) +
            ' ' + self.par.t_units)

        fig.write_image('output_figures/Fig6.png', scale=5.0)

    def plot_cumulative_figure(self):
        """Plots clustering output with Gaussians and threshols."""

        print('* Printing cumulative figure...')

        # Compute histogram of flattened self.data.matrix
        flat_m = self.data.matrix.flatten()
        counts, bins = np.histogram(flat_m, bins=self.par.bins, density=True)
        counts *= flat_m.size

        # Create a 1x2 subplots with shared y-axis
        fig, axes = plt.subplots(1, 2, sharey=True,
            gridspec_kw={'width_ratios': [3, 1]}, figsize=(9, 4.8))

        # Plot the histogram on the right subplot (axes[1])
        axes[1].stairs(counts, bins, fill=True,
            orientation='horizontal', alpha=0.5)

        # Create a color palette for plotting states
        palette = []
        n_states = len(self.states)
        cmap = plt.get_cmap('viridis', n_states + 1)
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Define time and y-axis limits for the left subplot (axes[0])
        y_spread = np.max(self.data.matrix) - np.min(self.data.matrix)
        y_lim = [np.min(self.data.matrix) - 0.025*y_spread,
            np.max(self.data.matrix) + 0.025*y_spread]
        time = self.par.print_time(self.data.matrix.shape[1])

        # Plot the individual trajectories on the left subplot (axes[0])
        step = 10 if self.data.matrix.size > 1000000 else 1
        for mol in self.data.matrix[::step]:
            axes[0].plot(time, mol, c='xkcd:black', ms=0.1, lw=0.1,
                alpha=0.5, rasterized=True)

        # Plot the Gaussian distributions of states on the right subplot (axes[1])
        for state_id, state in enumerate(self.states):
            popt = [state.mean, state.sigma, state.area]
            axes[1].plot(gaussian(np.linspace(bins[0], bins[-1], 1000), *popt),
                np.linspace(bins[0], bins[-1], 1000), color=palette[state_id])

        # Plot the horizontal lines and shaded regions to mark states' thresholds
        style_color_map = {
            0: ('--', 'xkcd:black'),
            1: ('--', 'xkcd:blue'),
            2: ('--', 'xkcd:red'),
        }

        time2 = np.linspace(time[0] - 0.05*(time[-1] - time[0]),
            time[-1] + 0.05*(time[-1] - time[0]), 100)
        for state_id, state in enumerate(self.states):
            linestyle, color = style_color_map.get(state.th_inf[1],
                ('-', 'xkcd:black'))
            axes[1].hlines(state.th_inf[0], xmin=0.0, xmax=np.amax(counts),
                linestyle=linestyle, color=color)
            axes[0].fill_between(time2, state.th_inf[0], state.th_sup[0],
                color=palette[state_id], alpha=0.25)
        axes[1].hlines(self.states[-1].th_sup[0], xmin=0.0, xmax=np.amax(counts),
            linestyle=linestyle, color='black')

        # Set plot titles and axis labels
        axes[0].set_ylabel('Signal')
        axes[0].set_xlabel(r'Time $t$ ' + self.par.t_units)
        axes[0].set_xlim([time2[0], time2[-1]])
        axes[0].set_ylim(y_lim)
        axes[1].set_xticklabels([])

        fig.savefig('output_figures/Fig2.png', dpi=600)
        plt.close(fig)

    def create_all_the_labels(self) -> np.ndarray:
        """
        Assigns labels to individual frames by repeating the existing labels.

        Returns:
        - np.ndarray: An updated ndarray with labels assigned
            to individual frames by repeating the existing labels.
        """
        all_the_labels = np.repeat(self.data.labels, self.par.tau_w, axis=1)
        return all_the_labels

    def plot_one_trajectory(self):
        """Plots the colored trajectory of an example particle."""

        example_id = self.par.example_id
        # Get the signal of the example particle
        all_the_labels = self.create_all_the_labels()
        signal = self.data.matrix[example_id][:all_the_labels.shape[1]]

        # Create time values for the x-axis
        time = self.par.print_time(all_the_labels.shape[1])

        # Create a figure and axes for the plot
        fig, axes = plt.subplots()

        unique_labels = np.unique(all_the_labels)
        # If there are no assigned window, we still need the "0" state
        # for consistency:
        if 0 not in unique_labels:
            unique_labels = np.insert(unique_labels, 0, 0)

        cmap = plt.get_cmap('viridis',
            np.max(unique_labels) - np.min(unique_labels) + 1)
        color = all_the_labels[example_id]
        axes.plot(time, signal, c='black', lw=0.1)

        axes.scatter(time, signal, c=color, cmap=cmap,
            vmin=np.min(unique_labels), vmax=np.max(unique_labels), s=1.0)

        # Add title and labels to the axes
        fig.suptitle('Example particle: ID = ' + str(example_id))
        axes.set_xlabel('Time ' + self.par.t_units)
        axes.set_ylabel('Signal')

        fig.savefig('output_figures/Fig3.png', dpi=600)
        plt.close(fig)

    def print_colored_trj_from_xyz(self, trj_file: str):
        """
        Creates a new XYZ file ('colored_trj.xyz') by coloring
        the original trajectory based on cluster labels.

        Args:
        - trj_file (str): Path to the original XYZ trajectory file.

        Steps:
        - Reads the original trajectory file 'trj_file'.
        - Removes the initial and final frames based on 'par.t_smooth',
            'par.t_delay', and available frames.
        - Creates a new XYZ file 'colored_trj.xyz' by adding cluster
            abels to the particle entries.
        """
        if os.path.exists(trj_file):
            print('* Loading trajectory.xyz...')
            with open(trj_file, "r", encoding="utf-8") as in_file:
                tmp = [line.strip().split() for line in in_file]

            all_the_labels = self.create_all_the_labels()
            num_of_particles = all_the_labels.shape[0]
            total_time = all_the_labels.shape[1]
            nlines = (num_of_particles + 2) * total_time

            frames_to_remove = int(self.par.t_smooth/2) + self.par.t_delay
            print('\t Removing the first', frames_to_remove, 'frames...')
            tmp = tmp[frames_to_remove * (num_of_particles + 2):]

            frames_to_remove = int((len(tmp) - nlines)/(num_of_particles + 2))
            print('\t Removing the last', frames_to_remove, 'frames...')
            tmp = tmp[:nlines]

            with open('colored_trj.xyz', "w+", encoding="utf-8") as out_file:
                i = 0
                for j in range(total_time):
                    print(tmp[i][0], file=out_file)
                    print('Properties=species:S:1:pos:R:3', file=out_file)
                    for k in range(num_of_particles):
                        print(all_the_labels[k][j],
                            tmp[i + 2 + k][1], tmp[i + 2 + k][2],
                            tmp[i + 2 + k][3], file=out_file)
                    i += num_of_particles + 2
        else:
            print('No ' + trj_file + ' found for coloring the trajectory.')

    def plot_tra_figure(self):
        """Plots time resolution analysis figures."""

        t_conv, units = self.par.t_conv, self.par.t_units
        min_t_s, max_t_s = self.par.min_t_smooth, self.par.max_t_smooth
        step_t_s = self.par.step_t_smooth

        time = self.number_of_states.T[0]*t_conv
        number_of_states = self.number_of_states[:, 1:].T
        fraction_0 = self.fraction_0[:, 1:].T

        for i, t_smooth in enumerate(range(min_t_s, max_t_s + 1, step_t_s)):
            fig, axes = plt.subplots()
            y_signal = number_of_states[i]
            y_2 = fraction_0[i]

            ### General plot settings ###
            axes.plot(time, y_signal, marker='o')
            axes.set_xlabel(r'Time resolution $\Delta t$ ' + units)#, weight='bold')
            axes.set_ylabel(r'# environments', weight='bold', c='#1f77b4')
            axes.set_xscale('log')
            axes.set_xlim(time[0]*0.75, time[-1]*1.5)
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))

            ### Top x-axes settings ###
            axes2 = axes.twiny()
            axes2.set_xlabel(r'Time resolution $\Delta t$ [frames]')
            axes2.set_xscale('log')
            axes2.set_xlim(time[0]*0.75/t_conv, time[-1]*1.5/t_conv)

            axesr = axes.twinx()
            axesr.plot(time, y_2, marker='o', c='#ff7f0e')
            axesr.set_ylabel('Population of env 0', weight='bold', c='#ff7f0e')

            fig.savefig('output_figures/Time_resolution_analysis_' +
                str(t_smooth) + '.png', dpi=600)

    def print_mol_labels_fbf_xyz(self):
        """
        Prints color IDs for Ovito visualization in XYZ format.

        Steps:
        - Creates a file ('all_cluster_IDs_xyz.dat') to store color IDs
            for Ovito visualization.
        - Iterates through each frame's molecular labels and writes
            them to the file in XYZ format.
        """
        print('* Print color IDs for Ovito...')
        all_the_labels = self.create_all_the_labels()
        with open('all_cluster_IDs_xyz.dat', 'w+', encoding="utf-8") as file:
            for j in range(all_the_labels.shape[1]):
                # Print two lines containing '#' to separate time steps.
                print('#', file=file)
                print('#', file=file)
                # Use np.savetxt to write the labels for each time step efficiently.
                np.savetxt(file, all_the_labels[:, j], fmt='%d', comments='')

    def print_mol_labels_fbf_gro(self):
        """
        Prints color IDs for Ovito visualization in GRO format.

        Args:
        - all_the_labels (np.ndarray): Array containing
            molecular labels for each frame.
        """
        print('* Print color IDs for Ovito...')
        all_the_labels = self.create_all_the_labels()
        with open('all_cluster_IDs_gro.dat', 'w', encoding="utf-8") as file:
            for labels in all_the_labels:
                print(' '.join(map(str, labels)), file=file)

    def print_mol_labels_fbf_lam(self):
        """
        Prints color IDs for Ovito visualization in .lammps format.

        Args:
        - all_the_labels (np.ndarray): Array containing
            molecular labels for each frame.
        """
        print('* Print color IDs for Ovito...')
        all_the_labels = self.create_all_the_labels()
        with open('all_cluster_IDs_lam.dat', 'w', encoding="utf-8") as file:
            for j in range(all_the_labels.shape[1]):
                # Print nine lines containing '#' to separate time steps.
                for _ in range(9):
                    print('#', file=file)
                np.savetxt(file, all_the_labels[:, j], fmt='%d', comments='')

    def print_signal_with_labels(self):
        """
        Creates a file ('signal_with_labels.dat') with signal values and associated cluster labels.

        Steps:
        - Checks the dimensionality of 'm_clean' to determine the signal attributes.
        - Writes the signals along with cluster labels for each frame to 'signal_with_labels.dat'.
        - Assumes the structure of 'm_clean' with signal values based on dimensionality (2D or 3D).
        - Incorporates 'all_the_labels' as cluster labels for respective frames.
        """
        m_clean = self.data.matrix
        all_the_labels = self.create_all_the_labels()

        with open('signal_with_labels.dat', 'w+', encoding="utf-8") as file:
            if m_clean.shape[2] == 2:
                print("Signal 1 Signal 2 Cluster Frame", file=file)
            else:
                print("Signal 1 Signal 2 Signal 3 Cluster Frame", file=file)
            for j in range(all_the_labels.shape[1]):
                for i in range(all_the_labels.shape[0]):
                    if m_clean.shape[2] == 2:
                        print(m_clean[i][j][0], m_clean[i][j][1],
                            all_the_labels[i][j], j + 1, file=file)
                    else:
                        print(m_clean[i][j][0], m_clean[i][j][1], m_clean[i][j][2],
                            all_the_labels[i][j], j + 1, file=file)
