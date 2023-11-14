import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import copy
import math
import scipy.optimize
from scipy.signal import savgol_filter
from scipy.signal import butter,filtfilt
from matplotlib.colors import LogNorm
import plotly
plotly.__version__
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

class State:
	def __init__(self, mu, sigma, A):
		number_of_sigmas = 2.0 							# The amplitude of the fluctiations INSIDE a state
		self.mu = mu 									# Mean of the Gaussian
		self.sigma = sigma 								# Variance of the Gaussian
		self.A = A 										# Area below the Gaussian
		self.peak = A/sigma/np.sqrt(np.pi)				# Height of the Gaussian peak
		self.perc = 0 									# Fraction of data points classified in this state
		self.th_inf = [mu - number_of_sigmas*sigma, -1]	# Lower thrashold of the state
		self.th_sup = [mu + number_of_sigmas*sigma, -1]	# Upper thrashold of the state

class State_multi_D:
	def __init__(self, mu, sigma, A):
		number_of_sigmas = 2.0 				# The amplitude of the fluctiations INSIDE a state
		self.mu = mu 						# Mean of the Gaussians
		self.sigma = sigma 					# Variance of the Gaussians
		self.A = A 							# Area below the Gaussians
		self.perc = 0 						# Fraction of data points classified in this state
		self.a = number_of_sigmas*sigma		# Axes of the state

class Parameters:
	def __init__(self, input_file):
		try:
			with open(input_file, 'r') as file:
				lines = file.readlines()
				param = [line.strip() for line in lines]
		except:
			print('\tinput_parameters.txt file missing or wrongly formatted.')
		if len(param) < 6:
			print('\tinput_parameters.txt file wrongly formatted.')

		self.tau_w = int(param[0])
		self.t_smooth = int(param[1])
		self.t_delay = int(param[2])
		self.t_conv = float(param[3])
		self.t_units = r'[' + str(param[4]) + r']'
		self.example_ID = int(param[5])
		self.bins = 'auto'

		if len(param) == 7:
			print('\tWARNING: overriding histogram binning')
			self.bins = int(param[6])
		elif len(param) > 7:
			print('\tinput_parameters.txt file wrongly formatted.')
