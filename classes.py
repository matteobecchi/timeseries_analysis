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
	def __init__(self, mu, sigma, area):
		number_of_sigmas = 2.0 							# The amplitude of the fluctiations INSIDE a state
		self.mu = mu 									# Mean of the Gaussian
		self.sigma = sigma 								# Variance of the Gaussian
		self.area = area 								# Area below the Gaussian
		self.peak = area/sigma/np.sqrt(np.pi)			# Height of the Gaussian peak
		self.perc = 0 									# Fraction of data points classified in this state
		self.th_inf = [mu - number_of_sigmas*sigma, -1]	# Lower thrashold of the state
		self.th_sup = [mu + number_of_sigmas*sigma, -1]	# Upper thrashold of the state

class State_multi:
	def __init__(self, mu, sigma, area):
		number_of_sigmas = 2.0 				# The amplitude of the fluctiations INSIDE a state
		self.mu = mu 						# Mean of the Gaussians
		self.sigma = sigma 					# Variance of the Gaussians
		self.area = area 					# Area below the Gaussians
		self.perc = 0 						# Fraction of data points classified in this state
		self.a = number_of_sigmas*sigma		# Axes of the state

class Parameters:
	def __init__(self, input_file):
		try:
			with open(input_file, 'r') as file:
				lines = file.readlines()
		except:
			print('\tinput_parameters.txt file missing or wrongly formatted.')
		if len(lines) < 6 or len(lines) > 7:
			print('\tinput_parameters.txt file wrongly formatted.')

		self.bins = 'auto'

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
				self.example_ID = int(value)
			if key == 'bins':
				self.bins = int(value)
				print('\tWARNING: overriding \'auto\' histogram binning')
