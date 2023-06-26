# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(tSOAP, LENS...). 

## Input data
A one-dimensional timeseries. 

## Input parameters
* `tau_window`: the amplidude of the time windows
* `tau_smooth`: sets the time resolution (Savgol_filter window)
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory
* `number_of_sigmas`: defines the width of the stable states
* `t_conv`: convert number of frames in time units