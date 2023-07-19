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
* `example_ID`: plots the trajectory of the molecule with this ID, colored according to the identified states. 

## Required Python 3 packages
`matplotlib`, `numpy`, `os`, `plotly`, `scipy`, `seaborn`, `sklearn`, `sys`. 

## Gaussian fitting procedure
1. The histogram of the timeseries is computed, using the `bins='auto'` numpy option. 
2. The histogram is smoothed with moving average with `window_size=3`. 
3. The absolute maximum of the histogram is found. 
4. Two Gaussian fits are performed:
 * The first one inside the interval between the two minima surrounding the maximum. 
 * The second one inside the interval where the peak around the maxima has its half height. 
5. Both fits, if converged, are evaluated according to the following points:
 * `mu` is contained inside the fit interval;
 * `sigma` is smaller than the fit interval;
 * the height of the peak is at least half the value of the maximum;
 * the relative uncertanty over the fit parameters is smaller than 0.5.
6. Finally, the fit with the best score is chosen. If only one of the two converged, that one is chosen. If none of the fits converges, the iterative procedure stops. 