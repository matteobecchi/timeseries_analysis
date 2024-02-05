"""
Contains the classes used for storing parameters and system states.
"""
import copy
from typing import Union
import scipy.signal
from matplotlib.colors import rgb2hex
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt

def read_from_xyz(data_file: str, col: int=4):
    with open(data_file, "r", encoding="utf-8") as file:
        tmp_list = [line.strip().split() for line in file]

    number_of_particles = int(tmp_list[0][0])
    number_of_frames = int(len(tmp_list) / (number_of_particles + 2))

    out_list = []
    line = 2
    for _ in range(number_of_frames):
        tmp = []
        for i in range(number_of_particles):
            tmp.append(tmp_list[line + i][col])
        out_list.append(tmp)
        line += number_of_particles + 2

    out_array = np.array(out_list, dtype=float).T
    return out_array

class StateUni:
    """
    Represents a state as a Gaussian.
    """
    def __init__(self, mean: float, sigma: float, area: float):
        self.mean = mean                                # Mean of the Gaussian
        self.sigma = sigma                              # Variance of the Gaussian
        self.area = area                                # Area below the Gaussian
        self.peak = area/sigma/np.sqrt(np.pi)           # Height of the Gaussian peak
        self.perc = 0                    # Fraction of data points classified in this state
        self.th_inf = [mean - 2.0*sigma, -1]    # Lower thrashold of the state
        self.th_sup = [mean + 2.0*sigma, -1]    # Upper thrashold of the state

    def build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds for the classification of data windows inside the state

        Args:
        - number of sigmas (float)
        """
        self.th_inf = [self.mean - number_of_sigmas*self.sigma, -1] # Lower thrashold of the state
        self.th_sup = [self.mean + number_of_sigmas*self.sigma, -1] # Upper thrashold of the state

class StateMulti:
    """
    Represents a state as a factorized Gaussian.
    """
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, area: np.ndarray):
        self.mean = mean         # Mean of the Gaussians
        self.sigma = sigma       # Variance of the Gaussians
        self.area = area         # Area below the Gaussians
        self.perc = 0            # Fraction of data points classified in this state
        self.axis = 2.0*sigma    # Axes of the state

    def build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds for the classification of data windows inside the state

        Args:
        - number of sigmas (float)
        """
        self.axis = number_of_sigmas*self.sigma     # Axes of the state

class UniData:
    """
    The input signals of the analysis.
    """
    def __init__(self, data_path: str):
        if data_path.endswith(('.npz', '.npy', '.txt', '.xyz')):
            try:
                if data_path.endswith('.npz'):
                    with np.load(data_path) as data:
                        # Load the first variable (assumed to be the data) into a NumPy array.
                        data_name = data.files[0]
                        self.matrix = np.array(data[data_name])
                elif data_path.endswith('.npy'):
                    self.matrix = np.load(data_path)
                elif data_path.endswith('.xyz'):
                    self.matrix = read_from_xyz(data_path)
                else: # .txt file
                    self.matrix = np.loadtxt(data_path, dtype=float)
                print('\tOriginal data shape:', self.matrix.shape)
            except Exception as exc_msg:
                print(f'\tERROR: Failed to read data from {data_path}. Reason: {exc_msg}')
                return
        else:
            print('\tERROR: unsupported format for input file.')
            return

        self.num_of_particles = self.matrix.shape[0]
        self.num_of_steps = self.matrix.shape[1]
        self.range = [ np.min(self.matrix), np.max(self.matrix) ]
        self.labels = np.array([])

    def print_info(self):
        """
        Prints information about the input data.
        """
        print('Number of particles:', self.num_of_particles)
        print('Number of steps:', self.num_of_steps)
        print('Data range:', self.range)

    def remove_delay(self, t_delay: int):
        """
        Removes a specified time delay from the data.

        Args:
        - t_delay (int): Number of steps to remove from the beginning of the data.
        """
        self.matrix = self.matrix[:, t_delay:]
        self.num_of_steps = self.matrix.shape[1]

    def smooth_lpf(self, sampling_freq: int, window: int):
        """
        Smooths the data using a digital low-passing, forward-backward filter.

        Args:
        - sampling_freq (int): the sampling frequency, in t_units
        - window (int): inverse of the maximum frequency (in frames).
        """
        if window == 1:
            return
        if window == 2:
            max_freq = sampling_freq/(window + 0.0000001)
        else:
            max_freq = sampling_freq/window
        coeff_1, coeff_0 =scipy.signal.iirfilter(4, Wn=max_freq, fs=sampling_freq,
            btype="low", ftype="butter")
        self.matrix = np.apply_along_axis(lambda x: scipy.signal.filtfilt(coeff_1, coeff_0, x),
            axis=1, arr=self.matrix)
        self.num_of_steps = self.matrix.shape[1]
        self.range = [ np.min(self.matrix), np.max(self.matrix) ]

    def smooth_mov_av(self, window: int):
        """
        Smooths the data using a moving average with a specified window size.

        Args:
        - window (int): Size of the moving average window.
        """
        weights = np.ones(window) / window
        self.matrix = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'),
            axis=1, arr=self.matrix)
        self.num_of_steps = self.matrix.shape[1]
        self.range = [ np.min(self.matrix), np.max(self.matrix) ]

    def normalize(self):
        """
        Normalizes the data between 0 and 1 based on its minimum and maximum values.
        """
        data_min, data_max = self.range[0], self.range[1]
        self.matrix = (self.matrix - data_min)/(data_max - data_min)
        self.range = [ np.min(self.matrix), np.max(self.matrix) ]

    def create_copy(self):
        """
        Returns an independent copy of the UniData object.
        Changes to the copy will not affect the original object.
        """
        copy_data = copy.deepcopy(self)
        return copy_data

    def plot_medoids(self, output_file: str):
        """
        Computes and plots the average time sequence inside each identified environment.

        Notes:
        - Computes cluster means and standard deviations for each identified cluster.
        - Plots the average time sequence and standard deviation for each cluster.
        - Saves the figure as a PNG file in the 'output_figures' directory.
        """
        tau_window = int(self.num_of_steps / self.labels.shape[1])
        all_the_labels = self.labels
        # Initialize lists to store cluster means and standard deviations
        center_list = []
        std_list = []

        # If there are no assigned window, we still need the "0" state
        # for consistency:
        missing_zero = 0
        list_of_labels = np.unique(all_the_labels)
        if 0 not in list_of_labels:
            list_of_labels = np.insert(list_of_labels, 0, 0)
            missing_zero = 1

        # Loop through unique labels (clusters)
        for ref_label in list_of_labels:
            tmp = []
            # Iterate through molecules and their labels
            for i, mol in enumerate(all_the_labels):
                for window, label in enumerate(mol):
                     # Define time interval
                    time_0 = window*tau_window
                    time_1 = (window + 1)*tau_window
                    # If the label matches the current cluster, append the corresponding data to tmp
                    if label == ref_label:
                        tmp.append(self.matrix[i][time_0:time_1])

            # Calculate mean and standard deviation for the current cluster
            if len(tmp) > 0:
                center_list.append(np.mean(tmp, axis=0))
                std_list.append(np.std(tmp, axis=0))

        # Create a color palette
        palette = []
        cmap = plt.get_cmap('viridis', list_of_labels.size)
        palette.append(rgb2hex(cmap(0)))
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Plot
        fig, ax = plt.subplots()
        time_seq = range(tau_window)
        for center_id, center in enumerate(center_list):
            err_inf = center - std_list[center_id]
            err_sup = center + std_list[center_id]
            ax.fill_between(time_seq, err_inf, err_sup, alpha=0.25,
                color=palette[center_id + missing_zero])
            ax.plot(time_seq, center, label='ENV'+str(center_id + missing_zero), marker='o',
                c=palette[center_id + missing_zero])
        fig.suptitle('Average time sequence inside each environments')
        ax.set_xlabel(r'Time $t$ [frames]')
        ax.set_ylabel(r'Signal')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

        fig.savefig('output_figures/' + output_file + '.png', dpi=600)

class MultiData:
    """
    The input signals of the analysis.
    """
    def __init__(self, path_list: str):
        data_list = []
        for data_path in path_list:
            if data_path.endswith(('.npz', '.npy', '.txt')):
                try:
                    if data_path.endswith('.npz'):
                        with np.load(data_path) as data:
                            # Load the first variable (assumed to be the data) into a NumPy array.
                            data_name = data.files[0]
                            data_list.append(np.array(data[data_name]))
                    elif data_path.endswith('.npy'):
                        data_list.append(np.load(data_path))
                    else: # .txt file
                        data_list.append(np.loadtxt(data_path))
                    print('\tOriginal data shape:', data_list[-1].shape)
                except Exception as exc_msg:
                    print(f'\tERROR: Failed to read data from {data_path}. Reason: {exc_msg}')
                    return
            else:
                print('\tERROR: unsupported format for input file.')
                return

        for dim, data in enumerate(data_list[:-1]):
            if data_list[dim].shape != data_list[dim + 1].shape :
                print('ERROR: The signals do not correspond. Abort.')
                # self.matrix = None
                return

        data_arr = np.array(data_list)
        self.matrix = np.transpose(data_arr, axes=(1, 2, 0))

        self.num_of_particles = self.matrix.shape[0]
        self.num_of_steps = self.matrix.shape[1]
        self.dims = self.matrix.shape[2]
        self.range = np.array([ [np.min(comp), np.max(comp)] for comp in data_list ])
        self.labels = np.array([])

    def print_info(self):
        """
        Prints information about the input data.
        """
        print('Number of particles:', self.num_of_particles)
        print('Number of steps:', self.num_of_steps)
        print('Number of components:', self.dims)
        print('Data range:', self.range)

    def remove_delay(self, t_delay: int):
        """
        Removes a specified time delay from the data.

        Args:
        - t_delay (int): Number of steps to remove from the beginning of the data.
        """
        self.matrix = self.matrix[:, t_delay:, :]
        self.num_of_steps = self.matrix.shape[1]

    def smooth(self, window: int):
        """
        Smooths the data using a moving average with a specified window size.

        Args:
        - window (int): Size of the moving average window.
        """
        weights = np.ones(window) / window
        tmp_matrix = np.transpose(self.matrix, axes=(2, 0, 1))
        tmp_matrix = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'),
            axis=2, arr=tmp_matrix)
        self.matrix = np.transpose(tmp_matrix, axes=(1, 2, 0))
        self.num_of_steps = self.matrix.shape[1]
        self.range = np.array([ [np.min(comp), np.max(comp)] for comp in tmp_matrix ])

    def normalize(self, dim_to_avoid: list[int]):
        """
        Normalizes the data between 0 and 1 based on its minimum and maximum values.
        """
        tmp_matrix = np.transpose(self.matrix, axes=(2, 0, 1))
        new_matrix = []
        for dim, comp in enumerate(tmp_matrix):
            if dim not in dim_to_avoid:
                data_min, data_max = np.min(comp), np.max(comp)
                new_matrix.append((comp - data_min)/(data_max - data_min))
            else:
                new_matrix.append(comp)
        self.matrix = np.transpose(np.array(new_matrix), axes=(1, 2, 0))
        self.range = np.array([ [np.min(comp), np.max(comp)] for comp in new_matrix ])

    def create_copy(self):
        """
        Returns an independent copy of the UniData object.
        Changes to the copy will not affect the original object.
        """
        copy_data = copy.deepcopy(self)
        return copy_data

    def plot_medoids(self, output_file: str):
        """
        Plot the mean time sequence for clusters in the data.

        Returns:
        - None: If the third dimension of input data is greater than 2.
        """
        if self.dims > 2:
            print('plot_medoids() does not work with 3D data.')
            return

        # Initialize lists to store cluster means and standard deviations
        tau_window = int(self.num_of_steps / self.labels.shape[1])
        center_list = []

        # Loop through unique labels (clusters)
        for ref_label in np.unique(self.labels):
            tmp = []
            # Iterate through molecules and their labels
            for i, mol in enumerate(self.labels):
                for j, label in enumerate(mol):
                    # Define time interval
                    t_0 = j*tau_window
                    t_1 = (j + 1)*tau_window
                    # If the label matches the current cluster, append the corresponding data to tmp
                    if label == ref_label:
                        tmp.append(self.matrix[i][t_0:t_1])

            # Calculate mean and standard deviation for the current cluster
            center_list.append(np.mean(tmp, axis=0))

        # Create a color palette
        palette = []
        cmap = plt.get_cmap('viridis', np.unique(self.labels).size)
        palette.append(rgb2hex(cmap(0)))
        for i in range(1, cmap.N):
            rgba = cmap(i)
            palette.append(rgb2hex(rgba))

        # Plot
        fig, ax = plt.subplots()
        for id_c, center in enumerate(center_list):
            sig_x = center[:, 0]
            sig_y = center[:, 1]
            ax.plot(sig_x, sig_y, label='ENV'+str(id_c), marker='o', c=palette[id_c])
        fig.suptitle('Average time sequence inside each environments')
        ax.set_xlabel(r'Signal 1')
        ax.set_ylabel(r'Signal 2')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

        fig.savefig('output_figures/' + output_file + '.png', dpi=600)

class Parameters:
    """
    Contains the set of parameters for the specific analysis.
    """
    def __init__(self, input_file: str):
        try:
            with open(input_file, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except:
            print('\tinput_parameters.txt file missing or wrongly formatted.')

        ### Ste the default values ###
        self.t_smooth = 1
        self.t_delay = 0
        self.t_conv = 1.
        self.t_units = '[frames]'
        self.example_id = 0
        self.bins: Union[str, int] = 'auto'
        self.num_tau_w = 20
        self.min_tau_w = 2
        self.max_tau_w = -1
        self.min_t_smooth = 1
        self.max_t_smooth = 5
        self.step_t_smooth = 1

        for line in lines:
            key, value = [ s for s in line.strip().split('\t') if s != '']
            if key == 'tau_window':
                self.tau_w = int(value)
            elif key == 't_smooth':
                self.t_smooth = int(value)
            elif key == 't_delay':
                self.t_delay = int(value)
            elif key == 't_conv':
                self.t_conv = float(value)
            elif key == 't_units':
                self.t_units = r'[' + str(value) + r']'
            elif key == 'example_ID':
                self.example_id = int(value)
            elif key == 'bins':
                self.bins = int(value)
            elif key == 'num_tau_w':
                self.num_tau_w = int(value)
            elif key == 'min_tau_w':
                self.min_tau_w = int(value)
            elif key == 'max_tau_w':
                self.max_tau_w = int(value)
            elif key == 'min_t_smooth':
                self.min_t_smooth = int(value)
            elif key == 'max_t_smooth':
                self.max_t_smooth = int(value)
            elif key == 'step_t_smooth':
                self.step_t_smooth = int(value)

    def print_time(self, num_of_steps: int):
        """
        Generates time values based on parameters and number of steps.

        Args:
        - num_of_steps (int): Number of time steps.

        Returns:
        - np.ndarray: Array of time values.
        """
        t_start = self.t_delay + int(self.t_smooth/2)
        time = np.linspace(t_start, t_start + num_of_steps, num_of_steps) * self.t_conv
        return time

    def print_to_screen(self):
        """
        Prints to screen all the analysis' parameters.
        """
        print('\n########################')
        print('### Input parameters ###')
        print('# tau_window = ', self.tau_w)
        print('# t_smooth = ', self.t_smooth)
        print('# t_delay = ', self.t_delay)
        print('# t_conv = ', self.t_conv)
        print('# t_units = ', self.t_units)
        print('# example_ID = ', self.example_id)
        print('# bins = ', self.bins)
        print('# num_tau_w = ', self.num_tau_w)
        print('# min_tau_w = ', self.min_tau_w)
        if self.max_tau_w > 1:
            print('# max_tau_w = ', self.max_tau_w)
        print('# min_t_smooth = ', self.min_t_smooth)
        print('# max_t_smooth = ', self.max_t_smooth)
        print('# step_t_smooth = ', self.step_t_smooth)
        print('########################\n')

    def create_copy(self):
        """
        Returns an independent copy of the Parameter object.
        Changes to the copy will not affect the original object.
        """
        copy_par = copy.deepcopy(self)
        return copy_par
