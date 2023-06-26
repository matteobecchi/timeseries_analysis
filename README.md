# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(*t*SOAP, LENS...). 

## Input data
A one-dimensional timeseries. Supported formats: .npy, npz.

## Input parameters
* `tau_window`: the amplidude of the time windows. 
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory. 
* `t_conv`: convert number of frames in time units. 
* `tau_smooth`: sets the time resolution (Savgol_filter window). 
* `number_of_sigmas`: defines the width of the stable states. 