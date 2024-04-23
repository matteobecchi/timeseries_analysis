# timeseries_analysis
Code for unsupervised clustering of time-series data. Reference to https://arxiv.org/abs/2402.07786 for further details. 

## Input data
A one-dimensional time-series, computed on *N* particles for *T* frames, in the format of a ndarray of shape *(N, T)*.

## Usage
Install the package using `pip install onion_clustering`. 

The `examples/` folder contains an example of usage. From this folder, download the example files as reported in the script `example_script_uni.py` and then run it with `python3 example_script_uni.py`. 

## input_parameters
* `tau_window` (int): the length of the time window (in number of frames). 
* `tau_window_list` (List[int], optional): the list of time_windows for which the number of states and the fraction of unclassified states will be measured.
* `bins` (int, optional): the number of bins used to compute histograms. This should be used only if all the fits fail with the automatic binning. 

## Output
The algorithm will attempt to perform the clustering on the input data, using different `tau_window` (geometrically spaced between 2 frames and the entire trajectory length, unless differently specified in the input parameters). 
The algorithm output consists in
* the list of Gaussian states characterizing the clusters;
* an array of shape *(N, T)* with the integer labels of the data points (unclassified data points are labelled with 0);
* an array of shape *(len(tau_window_list), 2)* with the number of states and the fraction of unclassified data points for every choice of the time resolution. 

## Multivariate time-series version
The multivariate time-series version of the algorithm works in a similar fashion, taking as input 2D or 3D data. The input array must have shape *(D, N, T)* where _D_ is the number of components. You can find an example of usage in `examples/example_script_multi.py`

## Required Python 3 packages
`matplotlib`, `numpy`, `plotly`, `scipy`. 

## Gaussian fitting procedure
1. The histogram of the time-series is computed, using the `bins='auto'` numpy option (unless a different `bins` is passed as imput parameter). 
2. The histogram is smoothed with moving average with `window_size=3` (unless there are less that 50 bins, in wich case no smoothing occurs). 
3. The absolute maximum of the histogram is found. 
4. Two Gaussian fits are performed:
 * The first one inside the interval between the two minima surrounding the maximum. 
 * The second one inside the interval where the peak around the maxima has its half height. 
5. Both fits, if converged, are evaluated according to the following points:
 * `mu` is contained inside the fit interval;
 * `sigma` is smaller than the fit interval;
 * the height of the peak is at least half the value of the maximum;
 * the relative uncertanty over the fit parameters is smaller than 0.5.
6. Finally, the fit with the best score is chosen. If only one of the two converged, that one is chosen. If none of the fits converges, the iterative procedure stops, returning a warning message. 

## Aknowledgements
Thanks to Andrew Tarzia for all the help with the code formatting and documentation, and to Domiziano Doria, Chiara Lionello and Simone Martino for the beta-testing. Writing all this code wouldn't have been possible without the help of ChatGPT. 
