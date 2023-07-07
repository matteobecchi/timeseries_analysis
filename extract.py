import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize
import scipy.stats
from scipy.signal import argrelextrema
import pycwt as wavelet
from functions import *

### System specific parameters ###
t_units = r'[ns]'			# Units of measure of time

### Usually no need to changhe these ###
output_file = 'states_output.txt'
poly_order = 2 				# Savgol filter polynomial order
n_bins = 100 				# Number of bins in the histograms
stop_th = 0.001				# Treshold to exit the maxima search

sankey_average = 10			# On how many frames to average the Sankey diagrams
show_plot = False			# Show all the plots

def all_the_input_stuff():
	### Read and clean the data points
	data_directory, PAR = read_input_parameters()
	if type(data_directory) == str:
		M_raw = read_data(data_directory)
	else:
		M0 = read_data(data_directory[0])
		M1 = read_data(data_directory[1])
		M_raw = np.array([ np.concatenate((M0[i], M1[i])) for i in range(len(M0)) ])
	
	with open('Cu211_300ps_06nm_LENS.txt', 'w') as f:
		for i, x in enumerate(M_raw):
			print('# Particle ' + str(i), file=f)
			for xt in x:
				print(xt, file=f)

	return M_raw

def main():
	M_raw = all_the_input_stuff()

if __name__ == "__main__":
	main()
