"""Auxiliary functions for plotting the results."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from onion_clustering.functions import gaussian

COLORMAP = "viridis"


def plot_output_uni(title: str, input_data: np.ndarray, state_list: List):
    """Plots clustering output with Gaussians and threshols."""
    # Compute histogram of flattened self.data.matrix
    flat_m = input_data.flatten()
    counts, bins = np.histogram(flat_m, bins=100, density=True)
    bins -= (bins[1] - bins[0]) / 2
    counts *= flat_m.size

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
    n_states = len(state_list)
    cmap = plt.get_cmap(COLORMAP, n_states + 1)
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Define time and y-axis limits for the left subplot (axes[0])
    y_spread = np.max(input_data) - np.min(input_data)
    y_lim = [
        np.min(input_data) - 0.025 * y_spread,
        np.max(input_data) + 0.025 * y_spread,
    ]
    t_steps = input_data.shape[1]
    time = np.linspace(0, t_steps - 1, t_steps)

    # Plot the individual trajectories on the left subplot (axes[0])
    step = 1
    if input_data.size > 1e6:
        step = 10
    for mol in input_data[::step]:
        axes[0].plot(
            time,
            mol,
            c="xkcd:black",
            ms=0.1,
            lw=0.1,
            alpha=0.5,
            rasterized=True,
        )

    for state_id, state in enumerate(state_list):
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
    for state_id, state in enumerate(state_list):
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
        state_list[-1].th_sup[0],
        xmin=0.0,
        xmax=np.amax(counts),
        linestyle=linestyle,
        color="black",
    )

    # Set plot titles and axis labels
    axes[0].set_ylabel("Signal")
    axes[0].set_xlabel(r"Time [frame]")
    axes[0].set_xlim([time2[0], time2[-1]])
    axes[0].set_ylim(y_lim)
    axes[1].set_xticklabels([])

    fig.savefig(title, dpi=600)


def plot_one_trj_uni(
    title: str, example_id: int, input_data: np.ndarray, labels: np.ndarray
):
    """Plots the colored trajectory of one example particle."""
    example_id = example_id
    signal = input_data[example_id][: labels.shape[1]]
    t_steps = labels.shape[1]
    time = np.linspace(0, t_steps - 1, t_steps)

    fig, axes = plt.subplots()
    unique_labels = np.unique(labels)
    # If there are no assigned window, we still need the "0" state
    # for consistency:
    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)

    cmap = plt.get_cmap(
        COLORMAP, np.max(unique_labels) - np.min(unique_labels) + 1
    )
    color = labels[example_id]
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
    fig.suptitle(f"Example particle: ID = {example_id}")
    axes.set_xlabel("Time [frame]")
    axes.set_ylabel("Signal")

    fig.savefig(title, dpi=600)


def plot_state_populations(title: str, labels: np.ndarray):
    """
    Plot the populations of states over time.

    - Initializes some useful variables
    - If all the points are classified, we still need the "0" state
        for consistency
    - Computes the population of each state at each time
    - Plots the results to Fig5.png

    """
    num_part = labels.shape[0]

    unique_labels = np.unique(labels)
    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)

    list_of_populations = []
    for label in unique_labels:
        population = np.sum(labels == label, axis=0)
        list_of_populations.append(population / num_part)

    palette = []
    n_states = unique_labels.size
    cmap = plt.get_cmap(COLORMAP, n_states)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    fig, axes = plt.subplots()
    t_steps = labels.shape[1]
    time = np.linspace(0, t_steps - 1, t_steps)
    for label, pop in enumerate(list_of_populations):
        axes.plot(time, pop, label=f"ENV{label}", color=palette[label])
    axes.set_xlabel(r"Time [frame]")
    axes.set_ylabel(r"Population")
    axes.legend()

    fig.savefig(title, dpi=600)


def plot_medoids_uni(
    title: str, tau_window: int, input_data: np.ndarray, labels: np.ndarray
):
    """
    Compute and plot the average signal sequence inside each state.

    - Initializes some usieful variables
    - Check if we need to add the "0" state for consistency
    - For each state, stores all the signal windows in that state, and
    - Computes mean and standard deviation of the signals in that state
    - Prints the output to file
    - Plots the results to Fig4.png

    """
    center_list = []
    std_list = []

    missing_zero = 0
    list_of_labels = np.unique(labels)
    if 0 not in list_of_labels:
        list_of_labels = np.insert(list_of_labels, 0, 0)
        missing_zero = 1

    for ref_label in list_of_labels:
        tmp = []
        for i, mol in enumerate(labels[:, ::tau_window]):
            for window, label in enumerate(mol):
                time_0 = window * tau_window
                time_1 = (window + 1) * tau_window
                if label == ref_label:
                    tmp.append(input_data[i][time_0:time_1])
        if len(tmp) > 0:
            center_list.append(np.mean(tmp, axis=0))
            std_list.append(np.std(tmp, axis=0))
    center_arr = np.array(center_list)
    std_arr = np.array(std_list)

    np.savetxt(
        "medoid_center.txt",
        center_arr,
        header="Signal average for each ENV",
    )
    np.savetxt(
        "medoid_stddev.txt",
        std_arr,
        header="Signal standard deviation for each ENV",
    )

    palette = []
    cmap = plt.get_cmap(COLORMAP, list_of_labels.size)
    palette.append(rgb2hex(cmap(0)))
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    fig, axes = plt.subplots()
    time_seq = range(tau_window)
    for center_id, center in enumerate(center_list):
        err_inf = center - std_list[center_id]
        err_sup = center + std_list[center_id]
        axes.fill_between(
            time_seq,
            err_inf,
            err_sup,
            alpha=0.25,
            color=palette[center_id + missing_zero],
        )
        axes.plot(
            time_seq,
            center,
            label=f"ENV{center_id + missing_zero}",
            marker="o",
            c=palette[center_id + missing_zero],
        )
    fig.suptitle("Average time sequence inside each environments")
    axes.set_xlabel(r"Time [frames]")
    axes.set_ylabel(r"Signal")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.legend()
    fig.savefig(title, dpi=600)


def plot_sankey(title: str, labels: np.ndarray, tmp_frame_list: list[int]):
    """
    Plots the Sankey diagram at the desired frames.

    Args:
    - tmp_frame_list (list[int]): list of frame indices

    - Initializes some useful variables
    - If there are no assigned window, we still need the "0" state
        for consistency
    - Create arrays to store the source, target, and value data
        for the Sankey diagram

    """
    all_the_labels = labels
    frame_list = np.array(tmp_frame_list)
    unique_labels = np.unique(all_the_labels)

    if 0 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, 0)
    n_states = unique_labels.size

    source = np.empty((frame_list.size - 1) * n_states**2)
    target = np.empty((frame_list.size - 1) * n_states**2)
    value = np.empty((frame_list.size - 1) * n_states**2)

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
    sankey_data = go.Sankey(link=link, node=node, arrangement="perpendicular")
    fig = go.Figure(sankey_data)

    # Add the title with the time information.
    fig.update_layout(title=f"Frames: {frame_list}")

    fig.write_image(title, scale=5.0)


def plot_time_res_analysis(title: str, tra: np.ndarray):
    """
    Plots the results of clustering at different time resolutions.

    Plots the number of states (including the unclassified points) and
    the fraction of unclassificed data points as a function of the time
    resolution used.
    """
    fig, axes = plt.subplots()
    axes.plot(tra[:, 0], tra[:, 1], marker="o")
    axes.set_xlabel(r"Time resolution $\Delta t$ [frame]")
    axes.set_ylabel(r"# environments", weight="bold", c="#1f77b4")
    axes.set_xscale("log")
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axesr = axes.twinx()
    axesr.plot(tra[:, 0], tra[:, 2], marker="o", c="#ff7f0e")
    axesr.set_ylabel("Population of env 0", weight="bold", c="#ff7f0e")
    fig.savefig(title, dpi=600)


def plot_medoids_multi(
    title: str, tau_window: int, input_data: np.ndarray, labels: np.ndarray
):
    """
    Compute and plot the average signal sequence inside each state.

    - Checks if the data have 2 or 3 components (works only with D == 2)
    - Initializes some usieful variables
    - Check if we need to add the "0" state for consistency
    - For each state, stores all the signal windows in that state, and
    - Computes mean of the signals in that state
    - Prints the output to file
    - Plots the results to Fig4.png

    """
    if input_data.shape[0] > 2:
        print("plot_medoids_multi() does not work with 3D data.")
        return

    missing_zero = 0
    list_of_labels = np.unique(labels)
    if 0 not in list_of_labels:
        list_of_labels = np.insert(list_of_labels, 0, 0)
        missing_zero = 1

    center_list = []

    for ref_label in list_of_labels:
        tmp = []
        for i, mol in enumerate(labels):
            for j, label in enumerate(mol):
                t_0 = j * tau_window
                t_1 = (j + 1) * tau_window
                if label == ref_label:
                    tmp.append(input_data.transpose(1, 2, 0)[i][t_0:t_1])
        center_list.append(np.mean(tmp, axis=0))

    palette = []
    cmap = plt.get_cmap(COLORMAP, np.unique(labels).size)
    palette.append(rgb2hex(cmap(0)))
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))
    fig, axes = plt.subplots()
    for id_c, center in enumerate(center_list):
        sig_x = center[:, 0]
        sig_y = center[:, 1]
        axes.plot(
            sig_x,
            sig_y,
            label=f"ENV{id_c + missing_zero}",
            marker="o",
            c=palette[id_c + missing_zero],
        )
    fig.suptitle("Average time sequence inside each environments")
    axes.set_xlabel(r"Signal 1")
    axes.set_ylabel(r"Signal 2")
    axes.legend()
    fig.savefig(title, dpi=600)


def plot_output_multi(
    title: str,
    input_data: np.ndarray,
    state_list: List,
    labels: np.ndarray,
    tau_window: int,
):
    """
    Plot a cumulative figure showing trajectories and identified states.
    """
    n_states = len(state_list) + 1
    tmp = plt.get_cmap(COLORMAP, n_states)
    colors_from_cmap = tmp(np.arange(0, 1, 1 / n_states))
    colors_from_cmap[-1] = tmp(1.0)
    m_clean = input_data.transpose(1, 2, 0)
    all_the_labels = np.repeat(labels, tau_window, axis=1)

    if m_clean.shape[2] == 3:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
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
                ax[a_0][a_1].plot(
                    mol.T[d_0],
                    mol.T[d_1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                color_list = all_the_labels[i * step]
                ax[a_0][a_1].scatter(
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
                ax[a_0][a_1].plot(
                    m_resized[id_min].T[d_0],
                    m_resized[id_min].T[d_1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                ax[a_0][a_1].scatter(
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
                ax[a_0][a_1].plot(
                    m_resized[id_max].T[d_0],
                    m_resized[id_max].T[d_1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                ax[a_0][a_1].scatter(
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
                    for state in state_list:
                        ellipse = Ellipse(
                            tuple(state.mean),
                            state.axis[d_0],
                            state.axis[d_1],
                            color="black",
                            fill=False,
                        )
                        ax[a_0][a_1].add_patch(ellipse)

            # Set plot titles and axis labels
            ax[a_0][a_1].set_xlabel(f"Signal {d_0}")
            ax[a_0][a_1].set_ylabel(f"Signal {d_1}")

        ax[1][1].axis("off")
        fig.savefig(title, dpi=600)
        plt.close(fig)

    elif m_clean.shape[2] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))

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
            ax.plot(
                mol.T[0],
                mol.T[1],
                c="black",
                lw=line_w,
                rasterized=True,
                zorder=0,
            )
            color_list = all_the_labels[i * step]
            ax.scatter(
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
        ax.plot(
            m_resized[id_min].T[0],
            m_resized[id_min].T[1],
            c="black",
            lw=line_w,
            rasterized=True,
            zorder=0,
        )
        ax.scatter(
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
        ax.plot(
            m_resized[id_max].T[0],
            m_resized[id_max].T[1],
            c="black",
            lw=line_w,
            rasterized=True,
            zorder=0,
        )
        ax.scatter(
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
        for state in state_list:
            ellipse = Ellipse(
                tuple(state.mean),
                state.axis[0],
                state.axis[1],
                color="black",
                fill=False,
            )
            ax.add_patch(ellipse)

        # Set plot titles and axis labels
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        fig.savefig(title, dpi=600)


def plot_one_trj_multi(
    title: str,
    example_id: int,
    tau_window: int,
    input_data: np.ndarray,
    labels: np.ndarray,
):
    """Plots the colored trajectory of an example particle."""

    m_clean = input_data.transpose(1, 2, 0)
    all_the_labels = np.repeat(labels, tau_window, axis=1)

    # Get the signal of the example particle
    sig_x = m_clean[example_id].T[0][: all_the_labels.shape[1]]
    sig_y = m_clean[example_id].T[1][: all_the_labels.shape[1]]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a colormap to map colors to the labels
    cmap = plt.get_cmap(
        COLORMAP,
        int(
            np.max(np.unique(all_the_labels))
            - np.min(np.unique(all_the_labels))
            + 1
        ),
    )
    color = all_the_labels[example_id]
    ax.plot(sig_x, sig_y, c="black", lw=0.1)

    ax.scatter(
        sig_x,
        sig_y,
        c=color,
        cmap=cmap,
        vmin=np.min(np.unique(all_the_labels)),
        vmax=np.max(np.unique(all_the_labels)),
        s=1.0,
        zorder=10,
    )

    # Set plot titles and axis labels
    fig.suptitle(f"Example particle: ID = {example_id}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    fig.savefig(title, dpi=600)
