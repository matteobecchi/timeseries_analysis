# timeseries_analysis
Code for identify dynamic enviroinments and events from a time series 
(tSOAP, LENS...). 

## Input data
A one-dimensional timeseries. 

## Input parameters
tau_smooth          sets the time resolution (Savgol_filter window)<br>
tau_delay           is for ignoring the first tau_delay frames of the trajectory<br>
number_of_sigmas    defines the width of the stable states<br>
