# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(*t*SOAP, LENS...). 

## Input data
A one-dimensional timeseries. Supported formats: .npy, npz.

## Usage
The working directory must contain:
* A text file called `input_parameters.txt` , whose format is esplained below
* A text file called `data_direcotry.txt` containing the path to the input data files
* A folder called `output_figures`, for the output figures 

From this directory, the code is run with `python3 ${PATH_TO_CODE}/main.py`. 

## Input parameters
* `tau_window`: the amplidude of the time windows. 
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory. 
* `t_conv`: convert number of frames in time units. 
* `tau_smooth`: sets the time resolution (Savgol_filter window). Default `tau_smooth = tau_window`. 
* `number_of_sigmas`: defines the width of the stable states. Default `number_of_sigmas = 2.0`. 