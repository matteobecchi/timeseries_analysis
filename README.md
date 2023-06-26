# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(_t_ SOAP, LENS...). 

## Input data
A one-dimensional timeseries. Supported formats: .npy, npz.

## Input parameters
* `tau_window`: the amplidude of the time windows
* `tau_smooth`: sets the time resolution (Savgol_filter window)
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory
* `number_of_sigmas`: defines the width of the stable states
* `t_conv`: convert number of frames in time units