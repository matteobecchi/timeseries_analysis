"""
Contains the classes used for storing the clustering data.
"""

import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from onion_clustering.first_classes import (
    MultiData,
    Parameters,
    StateMulti,
    StateUni,
    UniData,
)
from onion_clustering.functions import (
    gaussian,
    moving_average,
)

COLORMAP = "viridis"


class ClusteringObject:
    """This class contains input, output and methods for plotting."""

    def __init__(
        self,
        par: Parameters,
        data: Union[UniData, MultiData],
        number_of_sigmas: float,
    ):
        self.par = par
        self.number_of_sigmas = number_of_sigmas
        self.data = data
        self.iterations = -1
        self.tau_window_list: np.ndarray
        self.t_smooth_list: np.ndarray
        self.number_of_states: np.ndarray
        self.fraction_0: np.ndarray
        self.list_of_pop: List[List[List[float]]]

    def plot_input_data(self, filename: str):
        """Plots input data for visualization."""
        raise NotImplementedError

    def preparing_the_data(self):
        """Processes raw data for analysis."""
        raise NotImplementedError

    def plot_state_populations(self):
        """Plots the populations of states over time."""

        print("* Printing populations vs time...")
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
        cmap = plt.get_cmap(COLORMAP, n_states)
        for i in range(cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        fig, axes = plt.subplots()
        t_steps = self.data.labels.shape[1]
        time = self.par.print_time(t_steps)
        for label, pop in enumerate(list_of_populations):
            # pop_full = np.repeat(pop, self.par.tau_w)
            axes.plot(
                time, pop, label="ENV" + str(label), color=palette[label]
            )
        axes.set_xlabel(r"Time " + self.par.t_units)
        axes.set_ylabel(r"Population")
        axes.legend()

        fig.savefig("output_figures/Fig5.png", dpi=600)

    def sankey(self, tmp_frame_list: list[int]):
        """
        Computes and plots a Sankey diagram at the desired frames.

        Args:
        - tmp_frame_list (list[int]): List of frame indices.
        """
        print("* Computing and plotting the Sankey diagram...")

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

        tmp_label1 = []
        tmp_label2 = []

        # Loop through the frame_list and calculate the transition matrix
        # for each time window.
        for i, t_0 in enumerate(frame_list[:-1]):
            # Calculate the time jump for the current time window.
            t_jump = frame_list[i + 1] - frame_list[i]

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
                    tmp_label1.append(f"State {j}: {start_fr * 100:.2f}%")
                tmp_label2.append(f"State {j}: {end_fr * 100:.2f}%")

        arr_label1 = np.array(tmp_label1)
        arr_label2 = np.array(tmp_label2).flatten()

        # Concatenate the temporary labels to create the final node labels.
        label = np.concatenate((arr_label1, arr_label2))

        # Generate a color palette for the Sankey diagram.
        palette = []
        cmap = plt.get_cmap(COLORMAP, n_states)
        for i in range(cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Tile the color palette to match the number of frames.
        color = np.tile(palette, frame_list.size)

        # Create dictionaries to define the Sankey diagram nodes and links.
        node = {"label": label, "pad": 30, "thickness": 20, "color": color}
        link = {"source": source, "target": target, "value": value}

        # Create the Sankey diagram using Plotly.
        sankey_data = go.Sankey(
            link=link, node=node, arrangement="perpendicular"
        )
        fig = go.Figure(sankey_data)

        # Add the title with the time information.
        fig.update_layout(
            title="Frames: "
            + str(frame_list * self.par.t_conv)
            + " "
            + self.par.t_units
        )

        fig.write_image("output_figures/Fig6.png", scale=5.0)

    def create_all_the_labels(self) -> np.ndarray:
        """
        Assigns labels to individual frames by repeating the existing labels.

        Returns:
        - np.ndarray: An updated ndarray with labels assigned
            to individual frames by repeating the existing labels.
        """
        all_the_labels = np.repeat(self.data.labels, self.par.tau_w, axis=1)
        return all_the_labels

    def plot_cumulative_figure(self):
        """Plots clustering output with Gaussians and threshols."""
        raise NotImplementedError

    def plot_one_trajectory(self):
        """Plots the colored trajectory of an example particle."""
        raise NotImplementedError

    def print_colored_trj_from_xyz(self, trj_file: str):
        """
        Creates a new XYZ file ('colored_trj.xyz') by coloring
        the original trajectory based on cluster labels.

        Args:
        - trj_file (str): Path to the original XYZ trajectory file.
        """
        if os.path.exists(trj_file):
            print("* Loading trajectory.xyz...")
            with open(trj_file, "r", encoding="utf-8") as in_file:
                tmp = [line.strip().split() for line in in_file]

            all_the_labels = self.create_all_the_labels()
            num_of_particles = all_the_labels.shape[0]
            total_time = all_the_labels.shape[1]
            nlines = (num_of_particles + 2) * total_time

            frames_to_remove = int(self.par.t_smooth / 2) + self.par.t_delay
            print("\t Removing the first", frames_to_remove, "frames...")
            tmp = tmp[frames_to_remove * (num_of_particles + 2) :]

            frames_to_remove = int(
                (len(tmp) - nlines) / (num_of_particles + 2)
            )
            print("\t Removing the last", frames_to_remove, "frames...")
            tmp = tmp[:nlines]

            with open("colored_trj.xyz", "w+", encoding="utf-8") as out_file:
                i = 0
                for j in range(total_time):
                    print(tmp[i][0], file=out_file)
                    print("Properties=species:S:1:pos:R:3", file=out_file)
                    for k in range(num_of_particles):
                        print(
                            all_the_labels[k][j],
                            tmp[i + 2 + k][1],
                            tmp[i + 2 + k][2],
                            tmp[i + 2 + k][3],
                            file=out_file,
                        )
                    i += num_of_particles + 2
        else:
            print("No " + trj_file + " found for coloring the trajectory.")

    def print_labels(self):
        """
        Print the label for every particle for every frame.
        Output is a (N, T) array in a .npy file.
        """
        print("* Print labels for all the data points...")
        all_the_labels = self.create_all_the_labels()
        np.save("all_labels.npy", all_the_labels)

    def print_signal_with_labels(self):
        """
        Creates a file ('signal_with_labels.dat') with signal values
        and associated cluster labels.
        """
        m_clean = self.data.matrix
        all_the_labels = self.create_all_the_labels()

        with open("signal_with_labels.dat", "w+", encoding="utf-8") as file:
            if m_clean.shape[2] == 2:
                print("Signal 1 Signal 2 Cluster Frame", file=file)
            else:
                print("Signal 1 Signal 2 Signal 3 Cluster Frame", file=file)
            for j in range(all_the_labels.shape[1]):
                for i in range(all_the_labels.shape[0]):
                    if m_clean.shape[2] == 2:
                        print(
                            m_clean[i][j][0],
                            m_clean[i][j][1],
                            all_the_labels[i][j],
                            j + 1,
                            file=file,
                        )
                    else:
                        print(
                            m_clean[i][j][0],
                            m_clean[i][j][1],
                            m_clean[i][j][2],
                            all_the_labels[i][j],
                            j + 1,
                            file=file,
                        )

    def plot_tra_figure(self):
        """Plots time resolution analysis figures."""

        t_conv, units = self.par.t_conv, self.par.t_units
        min_t_s, max_t_s = self.par.min_t_smooth, self.par.max_t_smooth
        step_t_s = self.par.step_t_smooth

        time = self.number_of_states.T[0] * t_conv
        number_of_states = self.number_of_states[:, 1:].T
        fraction_0 = self.fraction_0[:, 1:].T

        for i, t_smooth in enumerate(range(min_t_s, max_t_s + 1, step_t_s)):
            fig, axes = plt.subplots()
            y_signal = number_of_states[i]
            y_2 = fraction_0[i]

            ### General plot settings ###
            axes.plot(time, y_signal, marker="o")
            axes.set_xlabel(r"Time resolution $\Delta t$ " + units)
            axes.set_ylabel(r"# environments", weight="bold", c="#1f77b4")
            axes.set_xscale("log")
            axes.set_xlim(time[0] * 0.75, time[-1] * 1.5)
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))

            ### Top x-axes settings ###
            axes2 = axes.twiny()
            axes2.set_xlabel(r"Time resolution $\Delta t$ [frames]")
            axes2.set_xscale("log")
            axes2.set_xlim(time[0] * 0.75 / t_conv, time[-1] * 1.5 / t_conv)

            axesr = axes.twinx()
            axesr.plot(time, y_2, marker="o", c="#ff7f0e")
            axesr.set_ylabel("Population of env 0", weight="bold", c="#ff7f0e")

            fig.savefig(
                "output_figures/Time_resolution_analysis_"
                + str(t_smooth)
                + ".png",
                dpi=600,
            )

    def plot_pop_fractions(self):
        """
        Plot, for every time resolution, the populations of the ENVs.

        The bottom state is the ENV0.
        """
        print("* Print populations fractions...")

        t_conv, units = self.par.t_conv, self.par.t_units
        time = self.tau_window_list * t_conv

        for i, t_smooth in enumerate(self.t_smooth_list):
            pop_array = self.list_of_pop[i]

            max_num_of_states = np.max(
                [len(pop_list) for pop_list in pop_array]
            )
            for _, pop_list in enumerate(pop_array):
                while len(pop_list) < max_num_of_states:
                    pop_list.append(0.0)

            pop_array = np.array(pop_array)

            fig, axes = plt.subplots()
            width = time[1:] - time[:-1]
            width = np.insert(width, 0, width[0] / 2)

            bottom = np.zeros(len(pop_array))
            for _, state in enumerate(pop_array.T):
                _ = axes.bar(
                    time, state, width, bottom=bottom, edgecolor="black"
                )
                bottom += state

            axes.set_xlabel(rf"Time resolution $\Delta t$ {units}")
            axes.set_ylabel(r"Population's fractions")
            axes.set_xscale("log")

            fig.savefig(
                f"output_figures/Populations_{t_smooth}.png",
                dpi=600,
            )


class ClusteringObject1D(ClusteringObject):
    """This class contains input, output and methods for plotting."""

    states: List[StateUni] = []

    def plot_input_data(self, filename: str):
        """Plots input data for visualization."""

        # Flatten the m_clean matrix and compute histogram counts and bins
        m_clean = self.data.matrix
        flat_m = m_clean.flatten()
        binning = self.par.bins
        counts, bins = np.histogram(flat_m, bins=binning, density=True)
        bins -= (bins[1] - bins[0]) / 2
        counts *= flat_m.size

        gap = 1
        if bins.size > 49:
            gap = int(bins.size * 0.02) * 2
        counts = moving_average(counts, gap)
        bins = moving_average(bins, gap)

        # Create a plot with two subplots (side-by-side)
        fig, axes = plt.subplots(
            1,
            2,
            sharey=True,
            gridspec_kw={"width_ratios": [3, 1]},
            figsize=(9, 4.8),
        )

        # Plot histogram in the second subplot (right side)
        axes[1].stairs(counts, bins, fill=True, orientation="horizontal")

        # Plot the individual trajectories in the first subplot (left side)
        time = self.par.print_time(m_clean.shape[1])
        step = 10 if m_clean.size > 1000000 else 1
        for mol in m_clean[::step]:
            axes[0].plot(
                time, mol, c="xkcd:black", lw=0.1, alpha=0.5, rasterized=True
            )

        # Set labels and titles for the plots
        axes[0].set_ylabel("Signal")
        axes[0].set_xlabel(r"Simulation time $t$ " + self.par.t_units)
        axes[1].set_xticklabels([])

        fig.savefig("output_figures/Fig0.png", dpi=600)
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
        print(
            "\tTrajectory has "
            + str(self.data.num_of_particles)
            + " particles. "
        )
        print(
            "\tTrajectory of length "
            + str(self.data.num_of_steps)
            + " frames ("
            + str(self.data.num_of_steps * t_conv),
            t_units + ")",
        )
        print(
            "\tUsing "
            + str(num_windows)
            + " windows of length "
            + str(tau_window)
            + " frames ("
            + str(tau_window * t_conv),
            t_units + ")",
        )

    def plot_cumulative_figure(self):
        """Plots clustering output with Gaussians and threshols."""

        print("* Printing cumulative figure...")

        # Compute histogram of flattened self.data.matrix
        flat_m = self.data.matrix.flatten()
        counts, bins = np.histogram(flat_m, bins=self.par.bins, density=True)
        bins -= (bins[1] - bins[0]) / 2
        counts *= flat_m.size

        gap = 1
        if bins.size > 49:
            gap = int(bins.size * 0.02) * 2
        counts = moving_average(counts, gap)
        bins = moving_average(bins, gap)

        # Create a 1x2 subplots with shared y-axis
        fig, axes = plt.subplots(
            1,
            2,
            sharey=True,
            gridspec_kw={"width_ratios": [3, 1]},
            figsize=(9, 4.8),
        )

        # Plot the histogram on the right subplot (axes[1])
        axes[1].stairs(
            counts, bins, fill=True, orientation="horizontal", alpha=0.5
        )

        # Create a color palette for plotting states
        palette = []
        n_states = len(self.states)
        cmap = plt.get_cmap(COLORMAP, n_states + 1)
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Define time and y-axis limits for the left subplot (axes[0])
        y_spread = np.max(self.data.matrix) - np.min(self.data.matrix)
        y_lim = [
            np.min(self.data.matrix) - 0.025 * y_spread,
            np.max(self.data.matrix) + 0.025 * y_spread,
        ]
        time = self.par.print_time(self.data.matrix.shape[1])

        # Plot the individual trajectories on the left subplot (axes[0])
        step = 10 if self.data.matrix.size > 1000000 else 1
        for mol in self.data.matrix[::step]:
            axes[0].plot(
                time,
                mol,
                c="xkcd:black",
                ms=0.1,
                lw=0.1,
                alpha=0.5,
                rasterized=True,
            )

        for state_id, state in enumerate(self.states):
            popt = [state.mean, state.sigma, state.area]
            axes[1].plot(
                gaussian(np.linspace(bins[0], bins[-1], 1000), *popt),
                np.linspace(bins[0], bins[-1], 1000),
                color=palette[state_id],
            )

        style_color_map = {
            0: ("--", "xkcd:black"),
            1: ("--", "xkcd:blue"),
            2: ("--", "xkcd:red"),
        }

        time2 = np.linspace(
            time[0] - 0.05 * (time[-1] - time[0]),
            time[-1] + 0.05 * (time[-1] - time[0]),
            100,
        )
        for state_id, state in enumerate(self.states):
            linestyle, color = style_color_map.get(
                state.th_inf[1], ("-", "xkcd:black")
            )
            axes[1].hlines(
                state.th_inf[0],
                xmin=0.0,
                xmax=np.amax(counts),
                linestyle=linestyle,
                color=color,
            )
            axes[0].fill_between(
                time2,
                state.th_inf[0],
                state.th_sup[0],
                color=palette[state_id],
                alpha=0.25,
            )
        axes[1].hlines(
            self.states[-1].th_sup[0],
            xmin=0.0,
            xmax=np.amax(counts),
            linestyle=linestyle,
            color="black",
        )

        # Set plot titles and axis labels
        axes[0].set_ylabel("Signal")
        axes[0].set_xlabel(r"Time $t$ " + self.par.t_units)
        axes[0].set_xlim([time2[0], time2[-1]])
        axes[0].set_ylim(y_lim)
        axes[1].set_xticklabels([])

        fig.savefig("output_figures/Fig2.png", dpi=600)
        plt.close(fig)

    def plot_one_trajectory(self):
        """Plots the colored trajectory of an example particle."""

        example_id = self.par.example_id
        # Get the signal of the example particle
        all_the_labels = self.create_all_the_labels()
        signal = self.data.matrix[example_id][: all_the_labels.shape[1]]

        # Create time values for the x-axis
        time = self.par.print_time(all_the_labels.shape[1])

        # Create a figure and axes for the plot
        fig, axes = plt.subplots()

        unique_labels = np.unique(all_the_labels)
        # If there are no assigned window, we still need the "0" state
        # for consistency:
        if 0 not in unique_labels:
            unique_labels = np.insert(unique_labels, 0, 0)

        cmap = plt.get_cmap(
            COLORMAP, np.max(unique_labels) - np.min(unique_labels) + 1
        )
        color = all_the_labels[example_id]
        axes.plot(time, signal, c="black", lw=0.1)

        axes.scatter(
            time,
            signal,
            c=color,
            cmap=cmap,
            vmin=np.min(unique_labels),
            vmax=np.max(unique_labels),
            s=1.0,
        )

        # Add title and labels to the axes
        fig.suptitle("Example particle: ID = " + str(example_id))
        axes.set_xlabel("Time " + self.par.t_units)
        axes.set_ylabel("Signal")

        fig.savefig("output_figures/Fig3.png", dpi=600)
        plt.close(fig)


class ClusteringObject2D(ClusteringObject):
    """This class contains input, output and methods for plotting."""

    states: List[StateMulti] = []

    def plot_input_data(self, filename: str):
        """
        Plot input data: histograms and trajectories.

        This function creates plots for input data:
        - For 2D data: Creates histograms and individual trajectories
            (side-by-side).
        - For 3D data: Creates a 3D plot showing the trajectories.
        """
        bin_selection = []
        counts_selection = []
        m_clean = self.data.matrix

        if isinstance(self.data, MultiData):
            for dim in range(self.data.dims):
                # Flatten the m matrix and compute histogram counts and bins
                flat_m = m_clean[:, :, dim].flatten()
                counts0, bins0 = np.histogram(
                    flat_m, bins=self.par.bins, density=True
                )
                counts0 *= flat_m.size
                bin_selection.append(bins0)
                counts_selection.append(counts0)

            if self.data.dims == 2:
                # Create a plot with two subplots (side-by-side)
                fig = plt.figure(figsize=(9, 9))
                grid = fig.add_gridspec(4, 4)
                ax1 = fig.add_subplot(grid[0:1, 0:3])
                ax2 = fig.add_subplot(grid[1:4, 0:3])
                ax3 = fig.add_subplot(grid[1:4, 3:4])
                ax1.set_xticklabels([])
                ax3.set_yticklabels([])

                # Plot histograms
                ax1.stairs(counts_selection[0], bin_selection[0], fill=True)
                ax3.stairs(
                    counts_selection[1],
                    bin_selection[1],
                    fill=True,
                    orientation="horizontal",
                )

                # Plot the individual trajectories in the first subplot
                id_max, id_min = 0, 0
                for idx, mol in enumerate(m_clean):
                    if np.max(mol) == np.max(m_clean):
                        id_max = idx
                    if np.min(mol) == np.min(m_clean):
                        id_min = idx
                step = 10 if m_clean.size > 1000000 else 1
                for idx, mol in enumerate(m_clean[::step]):
                    ax2.plot(
                        mol[:, 0],
                        mol[:, 1],
                        color="black",
                        lw=0.1,
                        alpha=0.5,
                        rasterized=True,
                    )
                ax2.plot(
                    m_clean[id_min][:, 0],
                    m_clean[id_min][:, 1],
                    color="black",
                    lw=0.1,
                    alpha=0.5,
                    rasterized=True,
                )
                ax2.plot(
                    m_clean[id_max][:, 0],
                    m_clean[id_max][:, 1],
                    color="black",
                    lw=0.1,
                    alpha=0.5,
                    rasterized=True,
                )

                # Set labels and titles for the plots
                ax2.set_ylabel("Signal 1")
                ax2.set_xlabel("Signal 2")

            elif self.data.dims == 3:
                fig = plt.figure(figsize=(6, 6))
                axes: Axes3D = fig.add_subplot(111, projection="3d")

                # Plot the individual trajectories
                step = 1 if m_clean.size > 1000000 else 1
                for idx, mol in enumerate(m_clean[::step]):
                    axes.plot(
                        mol[:, 0],
                        mol[:, 1],
                        mol[:, 2],
                        color="black",
                        marker="o",
                        ms=0.5,
                        lw=0.2,
                        alpha=1.0,
                        rasterized=True,
                    )

                # Set labels and titles for the plots
                axes.set_xlabel("Signal 1")
                axes.set_ylabel("Signal 2")
                axes.set_zlabel("Signal 3")

            fig.savefig("output_figures/" + filename + ".png", dpi=600)
            plt.close(fig)

    def preparing_the_data(self):
        """
        Prepare the raw data for analysis.

        This function prepares the raw data for analysis:
        - Applies a moving average filter on the raw data.
        - Normalizes the data to the range [0, 1] (commented out in the code).
        - Calculates the number of windows for analysis based on parameters.
        - Prints informative messages about trajectory details.
        """
        tau_window, t_smooth = self.par.tau_w, self.par.t_smooth
        t_conv, t_units = self.par.t_conv, self.par.t_units

        self.data.smooth(t_smooth)
        ### Normalizes data in [0, 1]. Usually not necessary.
        ### The arg is the list of components to NOT normalize
        # self.data.normalize([])

        # Calculate the number of windows for the analysis.
        num_windows = int(self.data.num_of_steps / tau_window)

        # Print informative messages about trajectory details.
        print(
            "\tTrajectory has "
            + str(self.data.num_of_particles)
            + " particles. "
        )
        print(
            "\tTrajectory of length "
            + str(self.data.num_of_steps)
            + " frames ("
            + str(self.data.num_of_steps * t_conv)
            + " "
            + t_units
            + ")."
        )
        print(
            "\tUsing "
            + str(num_windows)
            + " windows of length "
            + str(tau_window)
            + " frames ("
            + str(tau_window * t_conv)
            + " "
            + t_units
            + ")."
        )

    def plot_cumulative_figure(self):
        """
        Plot a cumulative figure showing trajectories and identified states.
        """
        print("* Printing cumulative figure...")
        n_states = len(self.states) + 1
        tmp = plt.get_cmap(COLORMAP, n_states)
        colors_from_cmap = tmp(np.arange(0, 1, 1 / n_states))
        colors_from_cmap[-1] = tmp(1.0)
        m_clean = self.data.matrix
        all_the_labels = self.create_all_the_labels()

        if m_clean.shape[2] == 3:
            fig, axes = plt.subplots(2, 2, figsize=(6, 6))
            dir0 = [0, 0, 1]
            dir1 = [1, 2, 2]
            ax0 = [0, 0, 1]
            ax1 = [0, 1, 0]

            for k in range(3):
                d_0 = dir0[k]
                d_1 = dir1[k]
                a_0 = ax0[k]
                a_1 = ax1[k]
                # Plot the individual trajectories
                id_max, id_min = 0, 0
                for idx, mol in enumerate(m_clean):
                    if np.max(mol) == np.max(m_clean):
                        id_max = idx
                    if np.min(mol) == np.min(m_clean):
                        id_min = idx

                line_w = 0.05
                max_t = all_the_labels.shape[1]
                m_resized = m_clean[:, :max_t:, :]
                step = 5 if m_resized.size > 1000000 else 1

                for i, mol in enumerate(m_resized[::step]):
                    axes[a_0][a_1].plot(
                        mol.T[d_0],
                        mol.T[d_1],
                        c="black",
                        lw=line_w,
                        rasterized=True,
                        zorder=0,
                    )
                    color_list = all_the_labels[i * step]
                    axes[a_0][a_1].scatter(
                        mol.T[d_0],
                        mol.T[d_1],
                        c=color_list,
                        cmap=COLORMAP,
                        vmin=0,
                        vmax=n_states - 1,
                        s=0.5,
                        rasterized=True,
                    )

                    color_list = all_the_labels[id_min]
                    axes[a_0][a_1].plot(
                        m_resized[id_min].T[d_0],
                        m_resized[id_min].T[d_1],
                        c="black",
                        lw=line_w,
                        rasterized=True,
                        zorder=0,
                    )
                    axes[a_0][a_1].scatter(
                        m_resized[id_min].T[d_0],
                        m_resized[id_min].T[d_1],
                        c=color_list,
                        cmap=COLORMAP,
                        vmin=0,
                        vmax=n_states - 1,
                        s=0.5,
                        rasterized=True,
                    )
                    color_list = all_the_labels[id_max]
                    axes[a_0][a_1].plot(
                        m_resized[id_max].T[d_0],
                        m_resized[id_max].T[d_1],
                        c="black",
                        lw=line_w,
                        rasterized=True,
                        zorder=0,
                    )
                    axes[a_0][a_1].scatter(
                        m_resized[id_max].T[d_0],
                        m_resized[id_max].T[d_1],
                        c=color_list,
                        cmap=COLORMAP,
                        vmin=0,
                        vmax=n_states - 1,
                        s=0.5,
                        rasterized=True,
                    )

                    # Plot the Gaussian distributions of states
                    if k == 0:
                        for state in self.states:
                            ellipse = Ellipse(
                                tuple(state.mean),
                                state.axis[d_0],
                                state.axis[d_1],
                                color="black",
                                fill=False,
                            )
                            axes[a_0][a_1].add_patch(ellipse)

                # Set plot titles and axis labels
                axes[a_0][a_1].set_xlabel(r"Signal " + str(d_0))
                axes[a_0][a_1].set_ylabel(r"Signal " + str(d_1))

            axes[1][1].axis("off")
            fig.savefig("output_figures/Fig2.png", dpi=600)
            plt.close(fig)

        elif m_clean.shape[2] == 2:
            fig, axes = plt.subplots(figsize=(6, 6))

            # Plot the individual trajectories
            id_max, id_min = 0, 0
            for idx, mol in enumerate(m_clean):
                if np.max(mol) == np.max(m_clean):
                    id_max = idx
                if np.min(mol) == np.min(m_clean):
                    id_min = idx

            line_w = 0.05
            max_t = all_the_labels.shape[1]
            m_resized = m_clean[:, :max_t:, :]
            step = 5 if m_resized.size > 1000000 else 1

            for i, mol in enumerate(m_resized[::step]):
                axes.plot(
                    mol.T[0],
                    mol.T[1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                color_list = all_the_labels[i * step]
                axes.scatter(
                    mol.T[0],
                    mol.T[1],
                    c=color_list,
                    cmap=COLORMAP,
                    vmin=0,
                    vmax=n_states - 1,
                    s=0.5,
                    rasterized=True,
                )

            color_list = all_the_labels[id_min]
            axes.plot(
                m_resized[id_min].T[0],
                m_resized[id_min].T[1],
                c="black",
                lw=line_w,
                rasterized=True,
                zorder=0,
            )
            axes.scatter(
                m_resized[id_min].T[0],
                m_resized[id_min].T[1],
                c=color_list,
                cmap=COLORMAP,
                vmin=0,
                vmax=n_states - 1,
                s=0.5,
                rasterized=True,
            )
            color_list = all_the_labels[id_max]
            axes.plot(
                m_resized[id_max].T[0],
                m_resized[id_max].T[1],
                c="black",
                lw=line_w,
                rasterized=True,
                zorder=0,
            )
            axes.scatter(
                m_resized[id_max].T[0],
                m_resized[id_max].T[1],
                c=color_list,
                cmap=COLORMAP,
                vmin=0,
                vmax=n_states - 1,
                s=0.5,
                rasterized=True,
            )

            # Plot the Gaussian distributions of states
            for state in self.states:
                ellipse = Ellipse(
                    tuple(state.mean),
                    state.axis[0],
                    state.axis[1],
                    color="black",
                    fill=False,
                )
                axes.add_patch(ellipse)

            # Set plot titles and axis labels
            axes.set_xlabel(r"$x$")
            axes.set_ylabel(r"$y$")

            fig.savefig("output_figures/Fig2.png", dpi=600)
            plt.close(fig)

    def plot_one_trajectory(self):
        """Plots the colored trajectory of an example particle."""

        m_clean = self.data.matrix
        all_the_labels = self.create_all_the_labels()

        # Get the signal of the example particle
        signal_x = m_clean[self.par.example_id].T[0][: all_the_labels.shape[1]]
        signal_y = m_clean[self.par.example_id].T[1][: all_the_labels.shape[1]]

        fig, axes = plt.subplots(figsize=(6, 6))

        # Create a colormap to map colors to the labels
        cmap = plt.get_cmap(
            COLORMAP,
            int(
                np.max(np.unique(all_the_labels))
                - np.min(np.unique(all_the_labels))
                + 1
            ),
        )
        color = all_the_labels[self.par.example_id]
        axes.plot(signal_x, signal_y, c="black", lw=0.1)

        axes.scatter(
            signal_x,
            signal_y,
            c=color,
            cmap=cmap,
            vmin=np.min(np.unique(all_the_labels)),
            vmax=np.max(np.unique(all_the_labels)),
            s=1.0,
            zorder=10,
        )

        # Set plot titles and axis labels
        fig.suptitle("Example particle: ID = " + str(self.par.example_id))
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$y$")

        fig.savefig("output_figures/Fig3.png", dpi=600)
        plt.close(fig)
