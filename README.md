# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(*t*SOAP, LENS...). 

## Input data
A one-dimensional timeseries. Supported formats: .npy, .npz.

## Usage
The working directory must contain:
* A text file called `input_parameters.txt` , whose format is explained below;
* A text file called `data_direcotry.txt` containing the path to the input data files.

From this directory, the code is run with `python3 ${PATH_TO_CODE}/main.py`. 

## input_parameters.txt
* `tau_window`: the amplidude of the time windows. 
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory. 
* `t_conv`: convert number of frames in time units. 
* `tau_smooth`: sets the time resolution (Savgol_filter window). Default choice is `tau_smooth = tau_window`. 
* `number_of_sigmas`: defines the width of the stable states. Default choice is `number_of_sigmas = 2.0`. 
* `example_ID`: plots the trajectory of the molecule with this ID, colored according to the identified states. 
* `resolution`: in computing the statistics, ignore windows shorter than `resolution*tau_windows` frames. 

## Required Python 3 packages
`matplotlib`, `numpy`, `os`, `plotly`, `scipy`, `seaborn`, `sklearn`, `sys`. 

## Gaussian fitting procedure
* The histogram of the signal is computed, and then smoothed with a moving average with width of 3 bins, to remove spurious local maxima and minima. 
* Maxima and minima are identified. 
* For every maximum, the fitting interval is identified as the intersection interval between the width at half height and the minima surrounding the maximum. 
* If the fitting interval is shorter than 4 bins, that maximum is discarded. 
* The Gaussian fit is then performed over a new histogram, computed only inside the fitting interval, in order to increase the binnin resolution in an adaptive way. 
* If the Gaussian average falls out of the fitting interval, that maximum is discarded. 
* If the Gaussian variance is larger that the fitting interval, that maximum is discarded. 
* If the relative uncertanty over one of the fit parameters is larger than 0.5, that maximum is discarded. 