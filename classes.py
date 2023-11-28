"""
Contains the classes used for storing parameters and system states.
"""

import numpy as np

class State:
    """
    Represents a state as a Gaussian.
    """
    def __init__(self, mean: float, sigma: float, area: float):
        number_of_sigmas = 2.0          # The amplitude of the fluctiations INSIDE a state
        self.mean = mean                                # Mean of the Gaussian
        self.sigma = sigma                              # Variance of the Gaussian
        self.area = area                                # Area below the Gaussian
        self.peak = area/sigma/np.sqrt(np.pi)           # Height of the Gaussian peak
        self.perc = 0                    # Fraction of data points classified in this state
        self.th_inf = [mean - number_of_sigmas*sigma, -1] # Lower thrashold of the state
        self.th_sup = [mean + number_of_sigmas*sigma, -1] # Upper thrashold of the state

class StateMulti:
    """
    Represents a state as a factorized Gaussian.
    """
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, area: np.ndarray):
        number_of_sigmas = 2.0              # The amplitude of the fluctiations INSIDE a state
        self.mean = mean                        # Mean of the Gaussians
        self.sigma = sigma                  # Variance of the Gaussians
        self.area = area                    # Area below the Gaussians
        self.perc = 0                       # Fraction of data points classified in this state
        self.axis = number_of_sigmas*sigma     # Axes of the state

class Parameters:
    """
    Contains the set of parameters for the specific analysis.
    """
    def __init__(self, input_file: str):
        try:
            with open(input_file, 'r') as file:
                lines = file.readlines()
        except:
            print('\tinput_parameters.txt file missing or wrongly formatted.')
        if len(lines) < 6 or len(lines) > 7:
            print('\tinput_parameters.txt file wrongly formatted.')

        self.t_smooth = 1
        self.t_delay = 1
        self.t_conv = 1.
        self.t_units = '[frames]'
        self.example_id = 0
        self.bins = 'auto'
        self.num_tau_w = 20
        self.min_tau_w = 2
        self.max_tau_w = -1
        self.min_t_smooth = 1
        self.max_t_smooth = 5

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
